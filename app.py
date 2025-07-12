import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import cv2
import os

# ------------------ UTILS ------------------ #
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_gradcam(model, img_array, last_conv_layer_name):
    # Build a model that outputs last conv layer and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(np.array(image), 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(superimposed)

# ------------------ LOAD MODELS ------------------ #
# (Assume files are included in the repository)
model_path = "plant_disease_model_mobilenetv2.h5"
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} not found. Please add it to the repository.")
    st.stop()
cnn_model = tf.keras.models.load_model(model_path)

rf_model_path = "yield_model.pkl"
if not os.path.exists(rf_model_path):
    st.error(f"Yield model file {rf_model_path} not found. Please add it to the repository.")
    st.stop()
rf_model = joblib.load(rf_model_path)

# Determine last conv layer for Grad-CAM
conv_layers = [layer.name for layer in cnn_model.layers 
               if isinstance(layer, tf.keras.layers.Conv2D)]
last_conv_layer = conv_layers[-1] if conv_layers else None

# Class names for the CNN model
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

# Crop labels for yield model
crop_labels = [
    'rice', 'maize', 'jute', 'cotton', 'coconut', 'banana', 'mango', 'grapes',
    'watermelon', 'orange', 'papaya', 'apple', 'muskmelon', 'lentil',
    'pomegranate', 'mothbeans', 'mungbean', 'blackgram', 'chickpea', 'coffee'
]

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(page_title="AI Crop Assistant", layout="wide")
st.title("🌿 AI-Powered Crop Assistant")

tab1, tab2 = st.tabs(["🦠 Disease Detection", "🌾 Yield Prediction"])

# ----- Tab 1: Disease Detection -----
with tab1:
    st.header("🦠 Leaf Disease Detection with Grad-CAM")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict disease class
        img_array = preprocess_image(image)
        prediction = cnn_model.predict(img_array)[0]
        class_idx = np.argmax(prediction)
        label = class_names[class_idx]
        st.success(f"✅ Prediction: {label}")

        # Grad-CAM visualization
        st.subheader("📊 Grad-CAM Heatmap")
        if last_conv_layer:
            try:
                heatmap = generate_gradcam(cnn_model, img_array, last_conv_layer)
                heatmap_img = overlay_heatmap(heatmap, image)
                st.image(heatmap_img, caption="Grad-CAM", use_column_width=True)
            except Exception as e:
                st.error(f"Grad-CAM Error: {e}")
        else:
            st.warning("No convolutional layer found for Grad-CAM.")

# ----- Tab 2: Yield Prediction -----
with tab2:
    st.header("🌾 Crop Yield Prediction + SHAP Explainability")

    with st.form("yield_form"):
        rainfall = st.number_input("Rainfall (mm)", value=100.0)
        temp = st.number_input("Temperature (°C)", value=25.0)
        humidity = st.number_input("Humidity (%)", value=60.0)
        ph = st.number_input("Soil pH", value=6.5)
        crop = st.selectbox("Crop Type", crop_labels)

        submitted = st.form_submit_button("Predict Yield")

        if submitted:
            crop_index = crop_labels.index(crop)
            input_data = pd.DataFrame([[rainfall, temp, humidity, ph, crop_index]],
                                      columns=["rainfall", "temp", "humidity", "ph", "crop_index"])
            prediction = rf_model.predict(input_data)[0]
            st.success(f"📈 Predicted Yield: {prediction:.2f} kg/hectare")

            # SHAP explainability
            st.subheader("🧠 SHAP Explanation")
            explainer = shap.Explainer(rf_model)
            shap_values = explainer(input_data)
            fig = shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

            # Downloadable report
            st.subheader("📄 Prediction Report")
            input_data["predicted_yield"] = prediction
            st.dataframe(input_data)
            csv = input_data.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", data=csv, file_name='prediction_report.csv', mime='text/csv')
