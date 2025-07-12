import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import joblib

# Import TensorFlow and suppress verbose logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Import SHAP for explanations
import shap

# Page configuration
st.set_page_config(page_title="AgriTech Assistant AI", layout="wide")

# Function to load models with caching
@st.cache_resource
def load_models():
    try:
        import gdown
    except ImportError:
        st.warning("Installing required package 'gdown'...")
        os.system('pip install gdown')
        import gdown

    # Load or download the disease detection model
    disease_model_path = "disease_model.h5"
    if not os.path.exists(disease_model_path):
        # Replace with your actual Google Drive shareable link
        disease_model_url = "https://drive.google.com/uc?id=YOUR_DISEASE_MODEL_ID"
        gdown.download(disease_model_url, disease_model_path, quiet=False)
    try:
        disease_model = tf.keras.models.load_model(disease_model_path)
    except Exception as e:
        st.error(f"Error loading disease model: {e}")
        disease_model = None

    # Load or download the yield prediction model
    yield_model_path = "yield_model.pkl"
    if not os.path.exists(yield_model_path):
        # Replace with your actual Google Drive shareable link
        yield_model_url = "https://drive.google.com/uc?id=YOUR_YIELD_MODEL_ID"
        gdown.download(yield_model_url, yield_model_path, quiet=False)
    try:
        yield_model = joblib.load(yield_model_path)
    except Exception as e:
        st.error(f"Error loading yield model: {e}")
        yield_model = None

    return disease_model, yield_model

# Load models
disease_model, yield_model = load_models()

# Disease detection class names (modify according to your model)
disease_class_names = [
    "Apple Scab", "Apple Black Rot", "Cedar Apple Rust",
    "Healthy Apple", "Healthy Grape", "Grape Black Rot"
]

# Utility functions for Grad-CAM
def get_img_array(img, size):
    array = np.array(img.resize(size))
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + tf.keras.backend.epsilon())
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    original = np.array(img)
    heatmap_color = cv2.resize(heatmap_color, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# Create two tabs for the app
tab1, tab2 = st.tabs(["Disease Detection & Grad-CAM", "Yield Prediction & SHAP"])

# -------- Tab 1: Disease Detection & Grad-CAM --------
with tab1:
    st.header("Crop Disease Detection and Grad-CAM")
    uploaded_file = st.file_uploader("Upload a crop leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Preprocess for model
            img_size = (224, 224)  # change if your model expects a different size
            img_array = get_img_array(image, size=img_size)
            # Make prediction
            if disease_model:
                preds = disease_model.predict(img_array)
                pred_index = np.argmax(preds[0])
                pred_label = (disease_class_names[pred_index] 
                              if pred_index < len(disease_class_names) 
                              else f"Class {pred_index}")
                st.success(f"Predicted Disease: **{pred_label}**")
                # Grad-CAM visualization
                last_conv_layer_name = 'block5_conv3'  # modify to your model's last conv layer
                try:
                    heatmap = make_gradcam_heatmap(img_array, disease_model, last_conv_layer_name, pred_index)
                    heatmap_image = overlay_heatmap(image, heatmap)
                    st.image(heatmap_image, caption="Grad-CAM Overlay", use_column_width=True)
                except Exception as e:
                    st.error(f"Could not generate Grad-CAM: {e}")
            else:
                st.error("Disease model is not loaded.")
        except Exception as e:
            st.error(f"Error processing image: {e}")

# -------- Tab 2: Yield Prediction & SHAP --------
with tab2:
    st.header("Crop Yield Prediction and Explanation")
    st.write("Enter the following features for yield prediction:")
    # Example features (adjust as needed)
    fertilizer = st.number_input("Fertilizer used (kg/ha)", value=0.0)
    rainfall = st.number_input("Annual Rainfall (mm)", value=0.0)
    temperature = st.number_input("Average Temperature (°C)", value=0.0)

    if st.button("Predict Yield"):
        features = np.array([[fertilizer, rainfall, temperature]])
        if yield_model:
            try:
                pred_yield = yield_model.predict(features)[0]
                st.success(f"Predicted Crop Yield: **{pred_yield:.2f} units**")
                # SHAP explanation
                st.write("SHAP values for the prediction:")
                try:
                    # Use SHAP Explainer for the model (waterfall plot for single prediction)
                    explainer = shap.Explainer(yield_model, np.zeros((1, features.shape[1])))
                    shap_values = explainer(features)
                    fig = shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig, bbox_inches='tight')
                except Exception as e:
                    st.error(f"Error generating SHAP values: {e}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("Yield model is not loaded.")
