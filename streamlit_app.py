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

    # yield prediction model (joblib pickle)
    yield_model_path = "yield_model.pkl"
    if not os.path.exists(yield_model_path):
        yield_model_url = "https://drive.google.com/uc?id=11kDYJhv-BAUj4sKmUby_QF9xGtu0NRPg"
        try:
            gdown.download(yield_model_url, yield_model_path, quiet=True)
        except Exception as e:
            st.error(f"Could not download yield model: {e}")

    yield_model = None
    if os.path.exists(yield_model_path):
        try:
            yield_model = joblib.load(yield_model_path)
            st.success("✅ Yield model loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading yield model: {e}")

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
    """Convert PIL image to model-ready numpy array."""
    img = img.convert('RGB')
    img_resized = img.resize(size)
    arr = np.array(img_resized).astype(np.float32)
    arr = preprocess_input(arr)
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
                raw_preds = disease_model.predict(img_array)

                if raw_preds.ndim == 2 and raw_preds.shape[1] > 1:
                    probs = tf.nn.softmax(raw_preds, axis=-1).numpy()[0]
                else:
                    probs = tf.nn.sigmoid(raw_preds).numpy().reshape(-1)

                if probs.shape[0] != len(class_names):
                    st.warning(
                        f"Warning: model returns {probs.shape[0]} classes but class_names has {len(class_names)} entries."
                    )

                pred_index = int(np.argmax(probs))
                confidence = float(probs[pred_index]) * 100.0
                pred_label = class_names[pred_index] if pred_index < len(class_names) else f"Class {pred_index}"

                st.success(f"Predicted Disease: **{pred_label}**")
                st.info(f"Confidence Score: **{confidence:.2f}%**")

                # Top-3 predictions
                top_k = min(3, probs.shape[0])
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in top_indices]
                top_scores = (probs[top_indices] * 100.0).tolist()

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

# -------- Tab 2: Yield Prediction & SHAP --------
with tab2:
    st.header("Crop Yield Prediction and Explanation")
    st.write("Enter features used by the yield model and get a predicted yield with SHAP explanation.")

    st.subheader("Input Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.number_input("Nitrogen (N)", value=0.0)
        P = st.number_input("Phosphorus (P)", value=0.0)
    with c2:
        K = st.number_input("Potassium (K)", value=0.0)
        rainfall = st.number_input("Annual Rainfall (mm)", value=0.0)
        temperature = st.number_input("Average Temperature (°C)", value=0.0)
    with c3:
        humidity = st.number_input("Average Humidity (%)", value=0.0)
        pH = st.number_input("Soil pH", value=7.0, min_value=0.0, max_value=14.0, step=0.1)

    if st.button("Predict Yield"):
        # ⚠️ FIX: removed undefined `fertilizer`
        features = np.array([[N, P, K, rainfall, temperature, humidity, pH]])

        if yield_model is None:
            st.error("Yield model is not loaded.")
        else:
            try:
                pred_yield = yield_model.predict(features)
                pred_val = float(pred_yield[0]) if hasattr(pred_yield, '__len__') else float(pred_yield)
                st.success(f"Predicted Crop Yield: **{pred_val:.2f}**")

                # SHAP explanation
                st.subheader("SHAP feature contributions")
                try:
                    explainer = shap.Explainer(yield_model, np.zeros((1, features.shape[1])))
                    shap_values = explainer(features)
                    sv = np.array(shap_values.values)
                    if sv.ndim == 3:
                        sv = sv[0]
                    shap_arr = sv[0] if sv.ndim == 2 and sv.shape[0] == 1 else sv

                    feature_names = ["N", "P", "K", "rainfall", "temperature", "humidity", "pH"]
                    df_shap = pd.DataFrame({"feature": feature_names, "shap_value": shap_arr})
                    df_shap = df_shap.sort_values(by='shap_value', key=lambda x: np.abs(x), ascending=False)

                    chart = alt.Chart(df_shap).mark_bar().encode(
                        x=alt.X('shap_value:Q', title='SHAP value (impact)'),
                        y=alt.Y('feature:N', sort='-x', title='Feature'),
                        tooltip=['feature', 'shap_value']
                    ).properties(height=250)
                    st.altair_chart(chart, use_container_width=True)

                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")

            except Exception as e:
                st.error(f"Error making prediction: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Notes: Make sure you scan your leaf for best results.")
