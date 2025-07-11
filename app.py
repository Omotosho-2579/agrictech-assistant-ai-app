import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import tempfile
import cv2
import gdown
from tensorflow.keras.models import Model

# Set TensorFlow to use only what's needed
tf.config.set_visible_devices([], 'GPU')  # Disable GPU in Hugging Face for compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages

# ------------------ UTILS ------------------ #
@st.cache_data
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input with caching"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@st.cache_data
def generate_gradcam(model, img_array, last_conv_layer_name="conv2d"):
    """Generate Grad-CAM heatmap with caching"""
    try:
        grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
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
    except Exception as e:
        st.error(f"Grad-CAM generation failed: {str(e)}")
        return None

@st.cache_data
def overlay_heatmap(heatmap, image, alpha=0.4):
    """Overlay heatmap on image with caching"""
    try:
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(np.array(image), 1 - alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(superimposed_img)
    except Exception as e:
        st.error(f"Heatmap overlay failed: {str(e)}")
        return image  # Return original image if overlay fails

def download_report(data):
    """Generate downloadable report without unsafe HTML"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return b64

# ------------------ MODEL LOADING WITH CACHING ------------------ #
@st.cache_resource
def load_cnn_model():
    """Load CNN model with caching and proper error handling"""
    model_path = "plant_disease_model_mobilenetv2.h5"
    gdown_url = "https://drive.google.com/uc?id=1fBVg3K3Tiu_TPb7JnT8gAonic4yHqhte"
    
    if not os.path.exists(model_path):
        try:
            with st.spinner("Downloading CNN model (200MB)..."):
                gdown.download(gdown_url, model_path, quiet=True)
        except Exception as e:
            st.error(f"Model download failed: {str(e)}")
            return None
    
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

@st.cache_resource
def load_rf_model():
    """Load Random Forest model with caching"""
    try:
        return joblib.load('yield_model.pkl')
    except Exception as e:
        st.error(f"Yield model loading failed: {str(e)}")
        return None

# ------------------ STREAMLIT UI ------------------ #
st.set_page_config(
    page_title="AI Crop Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models (this will trigger caching)
cnn_model = load_cnn_model()
rf_model = load_rf_model()

# Only proceed if models loaded successfully
if cnn_model is None or rf_model is None:
    st.error("⚠️ Critical error: Failed to load required models. Please check the logs.")
    st.stop()

st.title("🌿 AI-Powered Crop Assistant")
tabs = st.tabs(["🦠 Disease Detection", "🌾 Yield Prediction"])

# ------------------ TAB 1: Disease Detection ------------------ #
with tabs[0]:
    st.header("🦠 Leaf Disease Detection")
    
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

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing image..."):
                img_array = preprocess_image(image)
                prediction = cnn_model.predict(img_array)[0]
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction)
                label = class_names[class_idx]
                
                st.success(f"✅ Prediction: {label} (Confidence: {confidence:.2%})")

                # Grad-CAM Visualization
                st.subheader("📊 Model Attention Heatmap")
                heatmap = generate_gradcam(cnn_model, img_array)
                if heatmap is not None:
                    heatmap_img = overlay_heatmap(heatmap, image)
                    st.image(heatmap_img, caption="Model Attention Areas", use_column_width=True)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ------------------ TAB 2: Yield Prediction ------------------ #
with tabs[1]:
    st.header("🌾 Crop Yield Prediction")
    
    crop_labels = [
        'rice', 'maize', 'jute', 'cotton', 'coconut', 'banana', 'mango', 'grapes',
        'watermelon', 'orange', 'papaya', 'apple', 'muskmelon', 'lentil',
        'pomegranate', 'mothbeans', 'mungbean', 'blackgram', 'chickpea', 'coffee'
    ]

    with st.form("yield_form"):
        col1, col2 = st.columns(2)
        with col1:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=100.0)
            temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
        with col2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        
        crop = st.selectbox("Crop Type", crop_labels)
        submitted = st.form_submit_button("Predict Yield")

        if submitted:
            with st.spinner("Calculating prediction..."):
                try:
                    crop_index = crop_labels.index(crop)
                    input_data = pd.DataFrame([[rainfall, temp, humidity, ph, crop_index]],
                                            columns=["rainfall", "temp", "humidity", "ph", "crop_index"])
                    
                    prediction = rf_model.predict(input_data)[0]
                    st.success(f"📈 Predicted Yield: {prediction:.2f} kg/hectare")

                    # SHAP Explanation (with error handling)
                    try:
                        import shap
                        st.subheader("🧠 Prediction Explanation")
                        explainer = shap.Explainer(rf_model)
                        shap_values = explainer(input_data)
                        
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {str(e)}")

                    # Downloadable Report
                    st.subheader("📄 Prediction Report")
                    input_data["predicted_yield"] = prediction
                    st.dataframe(input_data)
                    
                    b64 = download_report(input_data)
                    st.download_button(
                        label="📥 Download Report",
                        data=base64.b64decode(b64),
                        file_name="crop_prediction_report.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)