---
title: AgriTech CCMT Assistant AI
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: true
license: mit
---

# 🌾 AgriTech CCMT Assistant AI

> **Professional AI-Powered Crop Disease Detection & Management System**  
> Specialized for Cashew, Cassava, Maize, and Tomato crops with 75.5% accuracy

An intelligent agricultural assistant that combines cutting-edge deep learning with practical farming insights to help farmers, agronomists, and agricultural researchers make data-driven decisions.

---

## ✨ Key Features

### 🔬 **Advanced Disease Detection**
- **22 Disease Classes** across 4 major crops (Cashew, Cassava, Maize, Tomato)
- **75.5% Validation Accuracy** with 97% Top-3 accuracy
- **MobileNetV2 Architecture** optimized for efficiency and accuracy
- **Real-time Image Quality Assessment** for better results
- **Grad-CAM Visualization** showing AI decision-making process

### 📊 **Smart Crop Recommendations**
- **Soil & Climate Analysis** for optimal crop selection
- **Multi-crop Yield Prediction** based on environmental parameters
- **SHAP Explainability** to understand prediction factors
- **Top-3 Recommendations** with confidence scores

### 💊 **Actionable Insights**
- **Treatment Recommendations** for detected diseases
- **Prevention Strategies** to minimize future outbreaks
- **Severity Assessment** (Low, Medium, High)
- **Professional Visual Reports** with confidence metrics

---

## 🎯 Supported Crops & Diseases

| Crop | Diseases Detected | Health Status |
|------|------------------|---------------|
| 🥜 **Cashew** | Anthracnose, Gumosis, Leaf Miner, Red Rust | ✅ Healthy |
| 🌿 **Cassava** | Bacterial Blight, Brown Spot, Green Mite, Mosaic | ✅ Healthy |
| 🌽 **Maize** | Fall Armyworm, Grasshopper, Leaf Beetle, Leaf Blight, Leaf Spot, Streak Virus | ✅ Healthy |
| 🍅 **Tomato** | Leaf Blight, Leaf Curl, Septoria Leaf Spot, Verticillium Wilt | ✅ Healthy |

**Total:** 22 disease classes with high-confidence detection

---

## 🚀 How to Use

### **Disease Detection Workflow**

1. **📸 Prepare Image**
   - Use clear, well-lit images
   - Focus on affected leaf areas
   - Natural daylight recommended
   - Minimum 800x800px resolution

2. **🔍 Upload & Analyze**
   - Upload JPG/PNG image
   - AI analyzes in 2-5 seconds
   - Automatic quality assessment

3. **📊 Review Results**
   - View predicted disease with confidence score
   - Check Top-5 alternative predictions
   - Examine Grad-CAM visual explanation

4. **💊 Take Action**
   - Read treatment recommendations
   - Implement prevention measures
   - Consult local experts if needed

### **Yield Prediction Workflow**

1. Enter soil nutrient levels (N, P, K)
2. Input climate conditions (temperature, humidity, rainfall)
3. Specify soil pH
4. Get AI-powered crop recommendations
5. Review SHAP feature importance

---

## 🧠 Technology Stack

### **Deep Learning**
- **Framework:** TensorFlow 2.15+ with Keras
- **Architecture:** MobileNetV2 (pre-trained on ImageNet)
- **Training Strategy:** Single-stage with partial unfreezing
- **Optimization:** Adam optimizer with learning rate scheduling
- **Data Augmentation:** Rotation, flipping, zoom, brightness adjustment

### **Machine Learning**
- **Yield Prediction:** Random Forest / Gradient Boosting
- **Feature Engineering:** Soil nutrients + climate factors
- **Explainability:** SHAP values for interpretability

### **Web Framework**
- **Frontend:** Streamlit with custom CSS
- **Visualization:** Plotly, Altair, Matplotlib
- **Image Processing:** PIL, OpenCV, NumPy

---

## 📈 Model Performance

### **Disease Detection (CCMT Model)**
```
✅ Validation Accuracy:     75.50%
✅ Top-3 Accuracy:          97.25%
✅ Top-5 Accuracy:          99.54%
✅ Training Samples:        20,110 images
✅ Validation Samples:      5,016 images
✅ Model Size:              71.48 MB
```

### **What This Means**
- **75.5% accuracy:** 3 out of 4 predictions are exactly correct
- **97% top-3 accuracy:** Correct disease is in top 3 predictions 97% of the time
- **99.5% top-5 accuracy:** Almost always in top 5 predictions

### **Confidence Levels**
- 🟢 **High (70-100%):** Reliable diagnosis - take action
- 🟡 **Medium (50-70%):** Likely accurate - verify if possible
- 🔴 **Low (<50%):** Uncertain - retake photo or consult expert

---

## 📦 Installation & Setup

### **Local Deployment**
```bash
# Clone repository
git clone https://github.com/yourusername/agritech-ccmt-ai.git
cd agritech-ccmt-ai

# Install dependencies
pip install -r requirements.txt

# Download models (or place manually)
# - ccmt_crop_disease_model.keras (71MB)
# - class_indices.json
# - yield_model.pkl

# Run app
streamlit run app.py
```

### **Cloud Deployment**

#### **Hugging Face Spaces** (Recommended)
1. Fork this repository
2. Upload model files to your space
3. App deploys automatically

#### **Streamlit Cloud**
1. Connect your GitHub repository
2. Add model files via Git LFS or Google Drive
3. Deploy with one click

---

## 📋 Requirements
```txt
streamlit==1.29.0
tensorflow==2.15.0
numpy==1.24.3
pandas==2.0.3
pillow==10.0.0
opencv-python==4.8.0
joblib==1.3.2
plotly==5.17.0
altair==5.1.2
scikit-learn==1.3.0
```

---

## 🔧 Configuration

### **Model Files Required**

| File | Size | Purpose | Required |
|------|------|---------|----------|
| `ccmt_crop_disease_model.keras` | 71 MB | Disease detection | ✅ Yes |
| `class_indices.json` | <1 KB | Class mapping | ✅ Yes |
| `yield_model.pkl` | Variable | Yield prediction | ⚠️ Optional |

### **For Large Files (>100MB)**
Use Git LFS or Google Drive with `gdown`:
```python
import gdown
if not os.path.exists('ccmt_crop_disease_model.keras'):
    gdown.download('YOUR_GOOGLE_DRIVE_LINK', 
                   'ccmt_crop_disease_model.keras', 
                   quiet=False)
```

---

## 📸 Usage Tips

### **For Best Disease Detection Results:**

✅ **DO:**
- Use natural daylight
- Capture affected leaf clearly
- Ensure image is in focus
- Fill frame with leaf
- Use high resolution (min 800x800px)

❌ **DON'T:**
- Use images with heavy shadows
- Include multiple overlapping leaves
- Upload blurry photos
- Use very small images
- Capture in low light conditions

---

## 🌍 Use Cases

- 🚜 **Small-Scale Farmers:** Quick disease diagnosis in the field
- 🎓 **Agricultural Extension:** Training and demonstration tool
- 🔬 **Researchers:** Data collection and analysis
- 🏢 **Agribusinesses:** Quality control and monitoring
- 📱 **Mobile Apps:** Integration for on-device diagnosis

---

## ⚠️ Limitations & Disclaimers

- **Not a Medical Tool:** This is an AI assistant, not a replacement for professional agricultural consultation
- **Crop Specific:** Only works for Cashew, Cassava, Maize, and Tomato
- **Best Effort:** 75.5% accuracy means ~1 in 4 predictions may be incorrect
- **Image Quality Dependent:** Poor image quality leads to poor predictions
- **Geographic Variations:** Disease manifestations may vary by region

**For severe infestations or uncertain diagnoses, always consult local agricultural experts.**

---

## 📊 Model Training Details

### **Dataset**
- **Name:** CCMT (Cashew, Cassava, Maize, Tomato)
- **Source:** Raw field images from African farms
- **Size:** 25,126 images (after cleaning)
- **Split:** 80% training, 20% validation
- **Augmentation:** Yes (rotation, flip, zoom, brightness)

### **Training Configuration**
- **Epochs:** 20 (with early stopping)
- **Batch Size:** 32
- **Learning Rate:** 0.0005 (with reduction on plateau)
- **Class Weights:** Enabled (handles imbalance)
- **Hardware:** GPU-accelerated training

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more crop types
- [ ] Improve accuracy with more training data
- [ ] Multi-language support
- [ ] PDF report generation
- [ ] Mobile app version
- [ ] Offline mode

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** CCMT Dataset from Mendeley Data
- **Architecture:** MobileNetV2 (Google)
- **Framework:** TensorFlow/Keras team
- **Inspiration:** Global food security and sustainable agriculture

---

## 📞 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/Omotosh-2579/agritech-assistant-ai-ap/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Omotosho-2579/agritech-assistant-ai-app/discussions)
- **Email:** abdulrafiumohammed2019@gmail.com

---

## 🌟 Citation

If you use this project in your research or work, please cite:
```bibtex
@software{agritech_ccmt_2024,
  title={AgriTech CCMT Assistant AI},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/spaces/yourusername/agritech-ccmt-ai}
}
```

---

## 🚀 Live Demo

👉 **[Try the App Now](https://huggingface.co/spaces/yourusername/agritech-ccmt-ai)** 🌾

---

<div align="center">

**Built with ❤️ for farmers and agricultural communities worldwide**

⭐ Star this repo if you find it useful! ⭐

</div>