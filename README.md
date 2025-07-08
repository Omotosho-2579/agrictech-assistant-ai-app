---
title: Farmer Assistant AI
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.26.0"
app_file: app.py
pinned: true
---

# 🌿 AI-Powered Crop Assistant (Streamlit App)

Welcome to the AI Crop Assistant — an intelligent tool for farmers, researchers, and agritech teams to:

- 🦠 Detect crop leaf diseases from images using a deep learning CNN (MobileNetV2)  
- 🌾 Predict expected crop yield based on soil and climate conditions using Random Forest  
- 📊 Explain predictions with SHAP (model explainability) and Grad-CAM (visual heatmaps)  
- 📥 Download prediction reports in CSV format  

---

## 📂 Features

| Feature                  | Details                                     |
|--------------------------|---------------------------------------------|
| Leaf disease detection   | Image classification using CNN              |
| Grad-CAM                 | Visual explanation of predictions           |
| Crop yield prediction    | Based on rainfall, temp, humidity, pH       |
| SHAP                     | Tabular model explainability                |
| Downloadable reports     | CSV summaries of yield predictions          |

---

## 🛠 How It Works

1. Upload a crop leaf image to detect disease  
2. Enter farming parameters (rainfall, temp, etc.)  
3. Select crop type and predict expected yield  
4. View explanation (SHAP) and Grad-CAM visual  
5. Download report 📄  

---

## 🧠 Technologies Used

- Streamlit  
- TensorFlow (CNN with MobileNetV2)  
- scikit-learn (Random Forest)  
- SHAP (Explainability)  
- OpenCV, Matplotlib, Pillow  

---

## 📦 Requirements

All dependencies are defined in `requirements.txt` and installed automatically by Hugging Face Spaces.

---

## 📁 Notes

- The CNN model (`.h5`) is downloaded via `gdown` from Google Drive  
- Make sure your file is publicly shared via Drive (or uploaded if <500MB)

---

## 🚀 Live App

👉 Click “Open in Spaces” above to use the app directly.
