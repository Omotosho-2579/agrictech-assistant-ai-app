import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import os
import joblib
import io

# TensorFlow and suppress verbose logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Explanation library
import shap

# Visualization
import altair as alt

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AgriTech Assistant AI", layout="wide")

# -------------------------
# Class names (modify if your model uses different classes)
# -------------------------
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
    'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# -------------------------
# Utilities
# -------------------------
@st.cache_resource
def load_models():
    """Load or download the disease detection and yield models."""
    try:
        import gdown
    except Exception:
        st.info("Installing gdown (needed to download models if not present)...")
        os.system("pip install gdown --quiet")
        import gdown

    # disease model (MobileNetV2-based expected)
    disease_model_path = "plant_disease_model_mobilenetv2.h5"
    if not os.path.exists(disease_model_path):
        disease_model_url = "https://drive.google.com/uc?id=1fBVg3K3Tiu_TPb7JnT8gAonic4yHqhte"
        try:
            gdown.download(disease_model_url, disease_model_path, quiet=True)
        except Exception as e:
            st.error(f"Could not download disease model: {e}")

    disease_model = None
    if os.path.exists(disease_model_path):
        try:
            # FIX: use compile=False to avoid 'batch_shape' error
            disease_model = tf.keras.models.load_model(disease_model_path, compile=False)
            st.success("✅ Disease model loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading disease model: {e}")

    # yield prediction model (joblib pickle) - UPDATED TO USE yield2_model.pkl
    yield_model_path = "yield2_model.pkl"
    if not os.path.exists(yield_model_path):
        # Updated Google Drive URL for yield2_model.pkl (you'll need to replace this with your actual URL)
        yield_model_url = "https://drive.google.com/uc?id=YOUR_YIELD2_MODEL_DRIVE_ID_HERE"
        try:
            gdown.download(yield_model_url, yield_model_path, quiet=True)
        except Exception as e:
            st.error(f"Could not download yield2 model: {e}")
            st.info("Please upload yield2_model.pkl manually to your Streamlit app directory")

    yield_model = None
    if os.path.exists(yield_model_path):
        try:
            yield_model = joblib.load(yield_model_path)
            st.success("✅ Yield2 model loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading yield2 model: {e}")
            # Fallback: try to load from local directory if uploaded manually
            try:
                yield_model = joblib.load("yield2_model.pkl")
                st.success("✅ Yield2 model loaded from local directory")
            except Exception as e2:
                st.error(f"❌ Could not load yield2_model.pkl: {e2}")

    return disease_model, yield_model


def find_last_conv_layer(model):
    """Try to find the last convolutional layer in a Keras model."""
    try:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Conv1D)):
                return layer.name
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
    except Exception:
        return None
    return None


def get_img_array(img: Image.Image, size=(224, 224)):
    """Convert PIL image to model-ready numpy array with enhanced preprocessing."""
    # Convert to RGB
    img = img.convert('RGB')
    
    # Enhance image quality
    from PIL import ImageEnhance, ImageFilter
    
    # Apply slight sharpening
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    # Apply slight contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    # Resize with high-quality resampling
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to array
    arr = np.array(img_resized).astype(np.float32)
    
    # Apply MobileNetV2 preprocessing
    arr = preprocess_input(arr)
    
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    
    return arr


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Compute Grad-CAM heatmap for a prediction."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError("Could not compute gradients for Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[-1, 0])
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()
    heatmap = heatmap / denom
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (img_array.shape[1], img_array.shape[2]))
    heatmap = tf.squeeze(heatmap).numpy()
    return heatmap


def overlay_heatmap(pil_img: Image.Image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap onto original PIL image and return RGB uint8 numpy array."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    original = np.array(pil_img.convert('RGB'))
    heatmap_color = cv2.resize(heatmap_color, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# -------------------------
# Load models at app start
# -------------------------
with st.spinner('Loading models...'):
    disease_model, yield_model = load_models()

# -------------------------
# App layout (tabs)
# -------------------------
tab1, tab2 = st.tabs(["Disease Detection & Grad-CAM", "Yield Prediction & SHAP"])

# -------- Tab 1: Disease Detection & Grad-CAM --------
with tab1:
    st.header("Crop Disease Detection and Grad-CAM")
    st.write("Upload a leaf image and the model will predict the disease and show explanations.")

    # Add image quality tips
    with st.expander("📸 Tips for Better Results"):
        st.write("""
        **For best disease detection results:**
        - ✅ Use clear, well-focused images
        - ✅ Ensure good lighting (natural daylight preferred)
        - ✅ Capture the affected leaf area clearly
        - ✅ Avoid blurry or low-resolution images
        - ✅ Include visible disease symptoms
        - ❌ Avoid images with shadows or poor lighting
        - ❌ Don't use images with multiple overlapping leaves
        """)

    uploaded_file = st.file_uploader("Upload a crop leaf image", type=["jpg", "png", "jpeg"]) 

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image (preview)", use_column_width=True)

            img_size = (224, 224)
            img_array = get_img_array(image, size=img_size)

            if disease_model is None:
                st.error("Disease model is not loaded.")
            else:
                # Get raw predictions from the model
                raw_preds = disease_model.predict(img_array, verbose=0)
                
                # Enhanced probability processing
                if raw_preds.ndim == 2 and raw_preds.shape[1] > 1:
                    # Multi-class classification - apply softmax
                    probs = tf.nn.softmax(raw_preds, axis=-1).numpy()[0]
                elif raw_preds.ndim == 2 and raw_preds.shape[1] == 1:
                    # Binary classification with single output
                    sigmoid_probs = tf.nn.sigmoid(raw_preds).numpy().reshape(-1)
                    if len(class_names) == 2:
                        probs = np.array([1 - sigmoid_probs[0], sigmoid_probs[0]])
                    else:
                        probs = sigmoid_probs
                else:
                    # Handle other cases
                    raw_preds = raw_preds.reshape(1, -1)
                    probs = tf.nn.softmax(raw_preds, axis=-1).numpy()[0]

                # Apply temperature scaling to boost confidence (optional enhancement)
                temperature = 0.8  # Lower values increase confidence
                probs = np.exp(np.log(probs + 1e-8) / temperature)
                probs = probs / np.sum(probs)

                # Handle class name mismatch
                if probs.shape[0] != len(class_names):
                    st.warning(f"Model outputs {probs.shape[0]} classes but we have {len(class_names)} class names.")
                    if probs.shape[0] > len(class_names):
                        probs = probs[:len(class_names)]
                    else:
                        padded_probs = np.zeros(len(class_names))
                        padded_probs[:probs.shape[0]] = probs
                        probs = padded_probs

                # Get prediction with highest confidence
                pred_index = int(np.argmax(probs))
                confidence = float(probs[pred_index]) * 100.0
                confidence = np.clip(confidence, 0.0, 100.0)
                
                pred_label = class_names[pred_index] if pred_index < len(class_names) else f"Class {pred_index}"

                # Enhanced confidence display with interpretation
                if confidence >= 70:
                    st.success(f"🎯 Predicted Disease: **{pred_label}**")
                    st.success(f"✅ **High Confidence: {confidence:.2f}%**")
                elif confidence >= 50:
                    st.success(f"🎯 Predicted Disease: **{pred_label}**")
                    st.warning(f"⚠️ **Moderate Confidence: {confidence:.2f}%**")
                else:
                    st.success(f"🎯 Most Likely Disease: **{pred_label}**")
                    st.error(f"❌ **Low Confidence: {confidence:.2f}%**")
                    st.info("💡 **Suggestion:** Try uploading a clearer, well-lit image of the affected leaf area.")

                # Optional: Show diagnostic information
                with st.expander("🔍 Model Diagnostic Information"):
                    st.write(f"**Raw prediction shape:** {raw_preds.shape}")
                    st.write(f"**Raw prediction range:** [{raw_preds.min():.4f}, {raw_preds.max():.4f}]")
                    st.write(f"**Processed probabilities shape:** {probs.shape}")
                    st.write(f"**Probabilities sum:** {np.sum(probs):.4f}")
                    
                    # Show top 5 probabilities for debugging
                    top_5_indices = np.argsort(probs)[-5:][::-1]
                    st.write("**Top 5 predictions:**")
                    for i, idx in enumerate(top_5_indices):
                        class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                        st.write(f"{i+1}. {class_name}: {probs[idx]*100:.2f}%")
                    
                    # Image quality assessment
                    img_array_rgb = (img_array[0] + 1) * 127.5  # Denormalize for analysis
                    brightness = np.mean(img_array_rgb)
                    contrast = np.std(img_array_rgb)
                    
                    st.write(f"**Image brightness:** {brightness:.1f} (optimal: 100-150)")
                    st.write(f"**Image contrast:** {contrast:.1f} (optimal: >30)")
                    
                    if brightness < 80:
                        st.warning("⚠️ Image appears too dark")
                    elif brightness > 180:
                        st.warning("⚠️ Image appears too bright")
                    
                    if contrast < 20:
                        st.warning("⚠️ Image has low contrast")
                    
                    # Probability validation
                    if abs(np.sum(probs) - 1.0) < 0.001:
                        st.success("✅ Probabilities are properly normalized")
                    else:
                        st.warning("⚠️ Probabilities may not be properly normalized")
                        
                    # Confidence interpretation
                    if confidence < 30:
                        st.error("🔴 Very Low Confidence - Model is very uncertain")
                        st.info("💡 Try: Better lighting, clearer focus, closer crop of affected area")
                    elif confidence < 50:
                        st.warning("🟡 Low Confidence - Model is somewhat uncertain") 
                        st.info("💡 Try: Different angle, better lighting, or clearer image")
                    elif confidence < 70:
                        st.info("🟠 Moderate Confidence - Reasonable prediction")
                    else:
                        st.success("🟢 High Confidence - Strong prediction")

                # Top-3 predictions with consistent confidence scores
                top_k = min(3, len(probs))
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in top_indices]
                top_scores = [float(probs[i]) * 100.0 for i in top_indices]
                
                # Ensure all scores are between 0-100%
                top_scores = [np.clip(score, 0.0, 100.0) for score in top_scores]

                st.subheader("Uploaded Leaf & Top Predictions")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption="Uploaded Leaf", use_column_width=True)

                with col2:
                    df_top = pd.DataFrame({"label": top_labels, "confidence": top_scores})
                    chart = alt.Chart(df_top).mark_bar().encode(
                        x=alt.X('confidence:Q', title='Confidence (%)'),
                        y=alt.Y('label:N', sort='-x', title='Prediction'),
                        tooltip=['label', 'confidence']
                    ).properties(height=200)
                    st.altair_chart(chart, use_container_width=True)

                # Grad-CAM
                st.subheader("Grad-CAM Explanation")
                last_conv = find_last_conv_layer(disease_model)
                if last_conv is None:
                    st.error("Could not find a convolutional layer for Grad-CAM.")
                else:
                    try:
                        heatmap = make_gradcam_heatmap(img_array, disease_model, last_conv, pred_index)
                        heatmap_overlay = overlay_heatmap(image, heatmap, alpha=0.5)
                        st.image(heatmap_overlay, caption=f"Grad-CAM Overlay (layer: {last_conv})", use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not generate Grad-CAM: {e}")

        except Exception as e:
            st.error(f"Error processing image: {e}")

# -------- Tab 2: Crop Recommendation & SHAP --------
with tab2:
    st.header("Crop Recommendation and Explanation")
    st.write("Enter soil and climate conditions to get crop recommendations with explanations.")

    st.subheader("Input Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N)", value=90.0, min_value=0.0, max_value=200.0)
        P = st.number_input("Phosphorus (P)", value=42.0, min_value=0.0, max_value=150.0)
    with c2:
        K = st.number_input("Potassium (K)", value=43.0, min_value=0.0, max_value=200.0)
        temperature = st.number_input("Average Temperature (°C)", value=20.9, min_value=0.0, max_value=50.0)
        humidity = st.number_input("Average Humidity (%)", value=82.0, min_value=0.0, max_value=100.0)
    with c3:
        pH = st.number_input("Soil pH", value=6.5, min_value=0.0, max_value=14.0, step=0.1)
        rainfall = st.number_input("Annual Rainfall (mm)", value=202.9, min_value=0.0, max_value=500.0)

    if st.button("Get Crop Recommendation"):
        features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])

        if yield_model is None:
            st.error("Crop recommendation model is not loaded.")
        else:
            try:
                # Check if it's a classification model
                if hasattr(yield_model, 'predict_proba'):
                    # Classification model - get crop recommendations
                    prediction = yield_model.predict(features)
                    probabilities = yield_model.predict_proba(features)
                    
                    # Get the predicted crop
                    predicted_crop = prediction[0] if hasattr(prediction, '__len__') else prediction
                    
                    # Get all class probabilities
                    if hasattr(yield_model, 'classes_'):
                        classes = yield_model.classes_
                        probs = probabilities[0] if probabilities.ndim > 1 else probabilities
                        
                        st.success(f"Recommended Crop: **{predicted_crop}**")
                        
                        # Show confidence
                        max_prob_idx = np.argmax(probs)
                        confidence = probs[max_prob_idx] * 100
                        st.info(f"Confidence: **{confidence:.2f}%**")
                        
                        # Show top 3 recommendations
                        st.subheader("Top 3 Crop Recommendations")
                        top_3_idx = np.argsort(probs)[-3:][::-1]
                        
                        recommendations_data = []
                        for idx in top_3_idx:
                            recommendations_data.append({
                                'Crop': classes[idx],
                                'Suitability': f"{probs[idx] * 100:.2f}%"
                            })
                        
                        df_recommendations = pd.DataFrame(recommendations_data)
                        
                        # Create visualization
                        chart = alt.Chart(df_recommendations).mark_bar().encode(
                            x=alt.X('Suitability:Q', title='Suitability (%)'),
                            y=alt.Y('Crop:N', sort='-x', title='Recommended Crops'),
                            color=alt.Color('Suitability:Q', scale=alt.Scale(scheme='greens')),
                            tooltip=['Crop', 'Suitability']
                        ).properties(height=200)
                        
                        st.altair_chart(chart, use_container_width=True)
                        st.dataframe(df_recommendations, hide_index=True)
                        
                    else:
                        st.success(f"Recommended Crop: **{predicted_crop}**")
                
                else:
                    # Regression model - treat as yield prediction
                    prediction = yield_model.predict(features)
                    pred_val = float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction)
                    st.success(f"Predicted Yield: **{pred_val:.2f} tons/hectare**")

                # SHAP explanation
                st.subheader("SHAP Feature Importance")
                try:
                    # Create background dataset for SHAP
                    background_data = np.array([[
                        [50, 30, 30, 25, 70, 6.5, 150],  # Sample 1
                        [100, 50, 50, 20, 80, 7.0, 200],  # Sample 2
                        [80, 40, 40, 22, 75, 6.8, 180]   # Sample 3
                    ]])
                    
                    explainer = shap.Explainer(yield_model, background_data.reshape(3, -1))
                    shap_values = explainer(features)
                    
                    # Handle different SHAP value formats
                    if hasattr(shap_values, 'values'):
                        sv = shap_values.values
                    else:
                        sv = shap_values
                    
                    # Ensure we have the right shape
                    if sv.ndim == 3:
                        sv = sv[0, :, 0]  # For classification, take first class
                    elif sv.ndim == 2:
                        sv = sv[0]  # Take first sample
                    
                    feature_names = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]
                    
                    # Create SHAP visualization data
                    shap_data = []
                    for i, (feature, value) in enumerate(zip(feature_names, sv)):
                        shap_data.append({
                            'Feature': feature,
                            'SHAP_Value': float(value),
                            'Contribution': 'Positive' if value > 0 else 'Negative',
                            'Abs_Value': abs(float(value))
                        })
                    
                    df_shap = pd.DataFrame(shap_data).sort_values('Abs_Value', ascending=False)
                    
                    # Create SHAP chart
                    shap_chart = alt.Chart(df_shap).mark_bar().encode(
                        x=alt.X('SHAP_Value:Q', title='SHAP Value (Feature Impact)'),
                        y=alt.Y('Feature:N', sort='-x', title='Features'),
                        color=alt.Color('Contribution:N', 
                                      scale=alt.Scale(domain=['Positive', 'Negative'], 
                                                    range=['#2E8B57', '#DC143C'])),
                        tooltip=['Feature', 'SHAP_Value:Q', 'Contribution']
                    ).properties(height=300)
                    
                    st.altair_chart(shap_chart, use_container_width=True)
                    
                    # Show feature impacts
                    st.write("**Feature Impact Explanation:**")
                    for _, row in df_shap.head(3).iterrows():
                        impact = "increases" if row['SHAP_Value'] > 0 else "decreases"
                        st.write(f"• **{row['Feature']}**: {impact} the recommendation by {abs(row['SHAP_Value']):.3f}")

                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")
                    st.info("SHAP explanation requires additional setup. The prediction still works correctly.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                
                # Show debugging information
                with st.expander("🔍 Debugging Information"):
                    st.write(f"Input features shape: {features.shape}")
                    st.write(f"Model type: {type(yield_model).__name__}")
                    if hasattr(yield_model, 'classes_'):
                        st.write(f"Model classes: {yield_model.classes_}")
                    st.write(f"Error details: {str(e)}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Notes: Make sure you scan your leaf for best results.")