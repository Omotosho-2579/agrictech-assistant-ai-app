import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import cv2
import os
import joblib
import json
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import base64

# TensorFlow and suppress verbose logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Visualization
import altair as alt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="AgriTech CCMT Assistant AI", 
    layout="wide",
    page_icon="🌾",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .disease-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
    h1 {
        color: #1E3A8A;
        font-weight: 700;
    }
    .stTab {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load CCMT Class Names
# -------------------------
@st.cache_data
def load_ccmt_class_names():
    """Load class names from the trained CCMT model"""
    class_indices_path = "class_indices.json"
    
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        # Reverse to get idx->name mapping
        idx_to_class = {v: k for k, v in class_indices.items()}
        return idx_to_class
    else:
        # Fallback: Default CCMT classes
        st.warning("class_indices.json not found. Using default CCMT classes.")
        return {
            0: 'Cashew_anthracnose', 1: 'Cashew_gumosis', 2: 'Cashew_healthy',
            3: 'Cashew_leaf miner', 4: 'Cashew_red rust',
            5: 'Cassava_bacterial blight', 6: 'Cassava_brown spot',
            7: 'Cassava_green mite', 8: 'Cassava_healthy', 9: 'Cassava_mosaic',
            10: 'Maize_fall armyworm', 11: 'Maize_grasshopper', 12: 'Maize_healthy',
            13: 'Maize_leaf beetle', 14: 'Maize_leaf blight',
            15: 'Maize_leaf spot', 16: 'Maize_streak virus',
            17: 'Tomato_healthy', 18: 'Tomato_leaf blight', 19: 'Tomato_leaf curl',
            20: 'Tomato_septoria leaf spot', 21: 'Tomato_verticillium wilt'
        }

# -------------------------
# Disease Information Database
# -------------------------
DISEASE_INFO = {
    'healthy': {
        'severity': 'None',
        'treatment': 'No treatment needed. Continue good agricultural practices.',
        'prevention': 'Maintain proper spacing, irrigation, and nutrient management.',
        'color': '#4CAF50'
    },
    'anthracnose': {
        'severity': 'High',
        'treatment': 'Apply copper-based fungicides. Remove infected plant parts.',
        'prevention': 'Use resistant varieties. Ensure proper drainage and air circulation.',
        'color': '#F44336'
    },
    'bacterial blight': {
        'severity': 'High',
        'treatment': 'Apply copper-based bactericides. Remove and destroy infected plants.',
        'prevention': 'Use disease-free planting material. Practice crop rotation.',
        'color': '#F44336'
    },
    'mosaic': {
        'severity': 'High',
        'treatment': 'No cure available. Remove infected plants to prevent spread.',
        'prevention': 'Control whitefly populations. Use virus-resistant varieties.',
        'color': '#F44336'
    },
    'leaf spot': {
        'severity': 'Medium',
        'treatment': 'Apply appropriate fungicides. Remove affected leaves.',
        'prevention': 'Avoid overhead irrigation. Ensure good air circulation.',
        'color': '#FF9800'
    },
    'rust': {
        'severity': 'Medium',
        'treatment': 'Apply fungicides containing sulfur or copper.',
        'prevention': 'Remove plant debris. Use resistant varieties.',
        'color': '#FF9800'
    },
    'blight': {
        'severity': 'High',
        'treatment': 'Apply systemic fungicides immediately. Improve drainage.',
        'prevention': 'Use certified disease-free seeds. Practice crop rotation.',
        'color': '#F44336'
    }
}

def get_disease_info(disease_name):
    """Extract disease information based on disease name"""
    disease_lower = disease_name.lower()
    
    if 'healthy' in disease_lower:
        return DISEASE_INFO['healthy']
    elif 'anthracnose' in disease_lower:
        return DISEASE_INFO['anthracnose']
    elif 'bacterial' in disease_lower or 'blight' in disease_lower:
        return DISEASE_INFO['bacterial blight']
    elif 'mosaic' in disease_lower:
        return DISEASE_INFO['mosaic']
    elif 'spot' in disease_lower:
        return DISEASE_INFO['leaf spot']
    elif 'rust' in disease_lower:
        return DISEASE_INFO['rust']
    else:
        return {
            'severity': 'Unknown',
            'treatment': 'Consult agricultural extension services for specific treatment.',
            'prevention': 'Practice good agricultural hygiene and monitoring.',
            'color': '#9E9E9E'
        }

# -------------------------
# Model Loading
# -------------------------
@st.cache_resource
def load_disease_model():
    """Load the trained CCMT MobileNetV2 model"""
    model_path = "ccmt_crop_disease_model.keras"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure 'ccmt_crop_disease_model.keras' is in the app directory")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile with appropriate metrics
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ CCMT Disease Detection Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def load_yield_model():
    """Load the yield prediction model"""
    yield_model_path = "yield2_model.pkl"
    
    if not os.path.exists(yield_model_path):
        st.warning("⚠️ Yield prediction model not found")
        return None
    
    try:
        yield_model = joblib.load(yield_model_path)
        st.success("✅ Yield Prediction Model loaded successfully")
        return yield_model
    except Exception as e:
        st.error(f"❌ Error loading yield model: {e}")
        return None

# -------------------------
# Image Processing
# -------------------------
def enhance_image(img: Image.Image):
    """Enhanced image preprocessing for better accuracy"""
    # Convert to RGB
    img = img.convert('RGB')
    
    # Apply enhancements
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    
    return img

def preprocess_image(img: Image.Image, size=(224, 224)):
    """Preprocess image for CCMT model"""
    # Enhance image
    img = enhance_image(img)
    
    # Resize with high quality
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(img_resized).astype(np.float32)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img_resized

def assess_image_quality(img: Image.Image):
    """Assess uploaded image quality"""
    img_array = np.array(img.convert('RGB'))
    
    # Calculate metrics
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    sharpness = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    
    quality_score = 0
    issues = []
    
    # Brightness check
    if 80 <= brightness <= 180:
        quality_score += 33
    else:
        if brightness < 80:
            issues.append("Image is too dark")
        else:
            issues.append("Image is too bright")
    
    # Contrast check
    if contrast > 25:
        quality_score += 33
    else:
        issues.append("Image has low contrast")
    
    # Sharpness check
    if sharpness > 100:
        quality_score += 34
    else:
        issues.append("Image appears blurry")
    
    return quality_score, issues, {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness
    }


# -------------------------
# Grad-CAM Visualization
# -------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap - handles nested MobileNetV2"""
    try:
        # For nested models like MobileNetV2, we need to access the base model
        base_model_layer = None
        
        # Find the MobileNetV2 base model
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                base_model_layer = layer
                break
        
        if base_model_layer is None:
            return None
        
        # Get the last convolutional layer from the base model
        last_conv_layer = None
        for layer in reversed(base_model_layer.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            return None
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay on original
    original_array = np.array(original_img)
    superimposed = cv2.addWeighted(original_array, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(superimposed)


# Grad-CAM helpers 

def list_conv_layer_names(model):
    """Return a list of Conv2D/DepthwiseConv2D layer names found in the model (including nested)."""
    names = []
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            names.append(layer.name)
        # nested model / functional block
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if isinstance(sub, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    names.append(sub.name)
    return names


def resolve_layer_output_tensor(model, layer_name):
    """
    Resolve a layer output tensor that is connected to the top-level model inputs.
    Returns the symbolic tensor or raises ValueError.
    """
    # 1) Try direct lookup on top-level model
    try:
        layer = model.get_layer(layer_name)
        # If this succeeds, layer.output should be connected to model.inputs
        return layer.output
    except Exception:
        pass

    # 2) Try searching nested submodels (common for MobileNetV2 wrapped as a layer)
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            try:
                sub = layer.get_layer(layer_name)
                # sub.output is symbolic within the nested model. But if the nested model instance
                # is the same object used in the top-level model, this tensor is typically usable
                # as an output in a top-level Model([...], [sub.output, model.output]).
                return sub.output
            except Exception:
                continue

    # 3) Not found in ways we can safely use
    raise ValueError(f"Could not find a layer output tensor for '{layer_name}' that is connected to the top-level model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Robust Grad-CAM heatmap generation.
    - img_array: (1, H, W, 3) float32 in [0,1]
    - model: top-level Keras Model
    - last_conv_layer_name: optional string. If None, auto-detect last conv layer.
    - pred_index: optional int class index to explain.
    Returns numpy heatmap (H, W) normalized to [0,1] or None.
    """
    try:
        # auto-detect last conv layer name if not provided
        if last_conv_layer_name is None:
            conv_names = list_conv_layer_names(model)
            if not conv_names:
                st.warning("Grad-CAM: no convolutional layers found in model.")
                return None
            last_conv_layer_name = conv_names[-1]  # take last by discovery order
            # helpful info for debugging
            st.info(f"Grad-CAM: auto-selected conv layer '{last_conv_layer_name}' from candidates: {conv_names}")

        # Resolve a symbolic tensor for the target conv layer that is connected to model.inputs
        try:
            target_conv_output = resolve_layer_output_tensor(model, last_conv_layer_name)
        except ValueError as e:
            st.warning(f"Grad-CAM: {e}")
            # show candidates to help user pick a layer manually
            st.warning("Available conv layer names: " + ", ".join(list_conv_layer_names(model)))
            return None

        # Build a model mapping model.inputs -> [target_conv_output, model.output]
        grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_conv_output, model.output])

        # Ensure img_array is a float32 tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            # Score for the target class
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            st.warning("Grad-CAM: gradients are None (unable to compute).")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]  # H x W x channels

        # Weighted combination
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize safely
        max_val = tf.math.reduce_max(heatmap)
        if max_val == 0:
            return tf.zeros_like(heatmap).numpy()
        heatmap = tf.maximum(heatmap, 0) / (max_val + 1e-10)

        return heatmap.numpy()

    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None


def overlay_heatmap(original_img, heatmap, alpha=0.4):
    """Overlay heatmap onto a PIL image and return a PIL image."""
    try:
        heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        original_array = np.array(original_img.convert('RGB'))
        superimposed = cv2.addWeighted(original_array, 1.0 - alpha, heatmap_colored, alpha, 0)
        return Image.fromarray(superimposed)
    except Exception as e:
        st.warning(f"Overlay heatmap failed: {e}")
        return original_img
# -------------------------
# PDF Report Generation
# -------------------------
def generate_pdf_report(disease_data, image_path=None):
    """Generate PDF report for disease detection"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🌾 AgriTech CCMT - Disease Detection Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Detection Results
    story.append(Paragraph("Detection Results", heading_style))
    
    results_data = [
        ['Parameter', 'Value'],
        ['Detected Disease', disease_data['disease']],
        ['Crop Type', disease_data['crop']],
        ['Confidence Level', f"{disease_data['confidence']:.1f}%"],
        ['Severity', disease_data['severity']],
    ]
    
    results_table = Table(results_data, colWidths=[3*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Treatment Recommendations
    story.append(Paragraph("Treatment Recommendations", heading_style))
    story.append(Paragraph(disease_data['treatment'], styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Prevention Measures
    story.append(Paragraph("Prevention Measures", heading_style))
    story.append(Paragraph(disease_data['prevention'], styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Top Predictions
    if 'top_predictions' in disease_data and disease_data['top_predictions']:
        story.append(Paragraph("Top 5 Alternative Diagnoses", heading_style))
        
        pred_data = [['Rank', 'Disease', 'Confidence']]
        for i, pred in enumerate(disease_data['top_predictions'], 1):
            pred_data.append([str(i), pred['Disease'], pred['Confidence']])
        
        pred_table = Table(pred_data, colWidths=[1*inch, 3.5*inch, 1.5*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(pred_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    story.append(Spacer(1, 0.4*inch))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_LEFT
    )
    story.append(Paragraph(
        "<b>Disclaimer:</b> This AI-powered diagnosis is provided as a decision support tool. "
        "For severe infestations or uncertain diagnoses, please consult local agricultural extension services or plant pathologists. "
        "Model accuracy: 75.5% (validation). Results may vary based on image quality and environmental factors.",
        disclaimer_style
    ))
    
    # Footer
    story.append(Spacer(1, 0.2*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER)
    story.append(Paragraph("Powered by AgriTech CCMT AI | MobileNetV2 Deep Learning Model", footer_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer
# -------------------------
# Main App
# -------------------------
def main():  
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/farm.png", width=80)
        st.title("🌾 AgriTech CCMT")
        st.markdown("### AI-Powered Crop Management")
        st.markdown("---")
        
        st.markdown("""
        **Supported Crops:**
        - 🥜 Cashew
        - 🌿 Cassava  
        - 🌽 Maize
        - 🍅 Tomato
        
        **Features:**
        - Disease Detection (22 classes)
        - AI Confidence Scoring
        - Visual Explanations (Grad-CAM)
        - Treatment Recommendations
        - Yield Prediction
        """)
        
        st.markdown("---")
        st.caption("Powered by MobileNetV2 & TensorFlow")
    
    # Header
    st.title("🌾 AgriTech Assistant AI")
    st.markdown("### Professional Crop Disease Detection & Management System")
    st.markdown("---")
    
    # Load models
    with st.spinner('🔄 Loading AI models...'):
        disease_model = load_disease_model()
        yield_model = load_yield_model()
        class_names = load_ccmt_class_names()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🔬 Disease Detection", 
        "📊 Yield Prediction",
        "📖 User Guide"
    ])
    
    # ========== TAB 1: DISEASE DETECTION ==========
    with tab1:
        st.header("🔬 Crop Disease Detection")
        st.markdown("Upload a clear image of the affected crop leaf for AI-powered disease detection")
        
        # Image quality tips
        with st.expander("📸 **Tips for Best Results**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **✅ DO:**
                - Use natural daylight
                - Capture affected leaf clearly
                - Ensure image is in focus
                - Fill frame with leaf
                - Use high resolution (min 800x800px)
                """)
            with col2:
                st.markdown("""
                **❌ DON'T:**
                - Use images with shadows
                - Include multiple overlapping leaves
                - Upload blurry photos
                - Use very small images
                - Capture in low light
                """)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Crop Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            try:
                # Load image
                original_image = Image.open(uploaded_file).convert('RGB')
                
                # Display original image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📷 Uploaded Image")
                    st.image(original_image, use_column_width=True)
                    
                    # Image quality assessment
                    quality_score, issues, metrics = assess_image_quality(original_image)
                    
                    st.markdown("#### Image Quality Assessment")
                    
                    # Quality score with color
                    if quality_score >= 80:
                        st.success(f"✅ Quality Score: {quality_score}/100 (Excellent)")
                    elif quality_score >= 60:
                        st.warning(f"⚠️ Quality Score: {quality_score}/100 (Good)")
                    else:
                        st.error(f"❌ Quality Score: {quality_score}/100 (Poor)")
                    
                    if issues:
                        st.warning("**Issues detected:**")
                        for issue in issues:
                            st.write(f"• {issue}")
                
                with col2:
                    if disease_model is None:
                        st.error("❌ Disease detection model not loaded")
                    else:
                        with st.spinner('🔄 Analyzing image...'):
                            # Preprocess
                            img_array, img_resized = preprocess_image(original_image)
                            
                            # Predict
                            predictions = disease_model.predict(img_array, verbose=0)
                            probs = predictions[0]
                            
                            # Get top prediction
                            pred_idx = int(np.argmax(probs))
                            confidence = float(probs[pred_idx]) * 100
                            pred_label = class_names.get(pred_idx, f"Class_{pred_idx}")
                            
                            # Extract crop and disease
                            if '_' in pred_label:
                                crop, disease = pred_label.split('_', 1)
                            else:
                                crop, disease = "Unknown", pred_label
                            
                            # Display results
                            st.subheader("🎯 Detection Results")
                            
                            # Confidence-based display
                            if confidence >= 70:
                                st.success(f"### ✅ {disease.replace('_', ' ').title()}")
                                conf_class = "confidence-high"
                                conf_emoji = "🟢"
                            elif confidence >= 50:
                                st.warning(f"### ⚠️ {disease.replace('_', ' ').title()}")
                                conf_class = "confidence-medium"
                                conf_emoji = "🟡"
                            else:
                                st.error(f"### ❌ {disease.replace('_', ' ').title()}")
                                conf_class = "confidence-low"
                                conf_emoji = "🔴"
                            
                            # Confidence display
                            st.markdown(f"""
                            <div style='background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                                <p style='margin:0; font-size: 0.9rem; color: #666;'>Confidence Level</p>
                                <p style='margin:0; font-size: 2rem; font-weight: bold;' class='{conf_class}'>
                                    {conf_emoji} {confidence:.1f}%
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**🌾 Crop:** {crop.title()}")
                            st.markdown(f"**🦠 Disease:** {disease.replace('_', ' ').title()}")
                            
                            # Get disease information
                            disease_info = get_disease_info(disease)
                            
                            st.markdown(f"""
                            <div style='background: {disease_info['color']}22; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {disease_info['color']}; margin: 1rem 0;'>
                                <strong>Severity:</strong> {disease_info['severity']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Treatment recommendations
                st.markdown("---")
                st.subheader("💊 Treatment & Prevention")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🩺 Recommended Treatment")
                    st.info(disease_info['treatment'])
                
                with col2:
                    st.markdown("#### 🛡️ Prevention Measures")
                    st.info(disease_info['prevention'])
                
                # Top 5 predictions
                st.markdown("---")
                st.subheader("📊 Detailed Analysis")
                
                top_k = min(5, len(probs))
                top_indices = np.argsort(probs)[-top_k:][::-1]
                
                # Create dataframe
                top_predictions = []
                for idx in top_indices:
                    label = class_names.get(idx, f"Class_{idx}")
                    prob = float(probs[idx]) * 100
                    top_predictions.append({
                        'Rank': len(top_predictions) + 1,
                        'Disease': label.replace('_', ' ').title(),
                        'Confidence': f"{prob:.2f}%",
                        'Probability': prob
                    })
                
                df_top = pd.DataFrame(top_predictions)
                
                # Plotly chart
                fig = go.Figure(data=[
                    go.Bar(
                        y=df_top['Disease'],
                        x=df_top['Probability'],
                        orientation='h',
                        marker=dict(
                            color=df_top['Probability'],
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Confidence %")
                        ),
                        text=df_top['Confidence'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Top 5 Disease Predictions",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Disease",
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Grad-CAM visualization
                st.markdown("---")
                st.subheader("🔍 AI Visual Explanation (Grad-CAM)")
                st.markdown("Highlighted regions show where the AI focused to make its decision")
                
                last_conv = find_last_conv_layer(disease_model)
                
                if last_conv:
                    with st.spinner('Generating visual explanation...'):
                        heatmap = make_gradcam_heatmap(img_array, disease_model, last_conv, pred_idx)
                        
                        if heatmap is not None:
                            overlay_img = overlay_heatmap(original_image, heatmap, alpha=0.5)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(original_image, caption="Original Image", use_column_width=True)
                            with col2:
                                st.image(overlay_img, caption="Grad-CAM Overlay (Focus Areas in Red)", use_column_width=True)
                                
                            # Store overlay for PDF
                            st.session_state['gradcam_overlay'] = overlay_img
                        else:
                            st.warning("Could not generate Grad-CAM visualization")
                else:
                    st.warning("Grad-CAM not available for this model architecture")
                
                # Grad-CAM visualization
                st.markdown("---")
                st.subheader("🔍 AI Visual Explanation (Grad-CAM)")
                st.markdown("Highlighted regions show where the AI focused to make its decision")
                
                last_conv = find_last_conv_layer(disease_model) if disease_model is not None else None
                
                if last_conv:
                    with st.spinner('Generating visual explanation...'):
                        heatmap = make_gradcam_heatmap(img_array, disease_model, last_conv, pred_idx)
                        
                        if heatmap is not None:
                            overlay_img = overlay_heatmap(original_image, heatmap, alpha=0.5)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(original_image, caption="Original Image", use_column_width=True)
                            with col2:
                                st.image(overlay_img, caption="Grad-CAM Overlay (Focus Areas in Red)", use_column_width=True)
                            
                            # Store overlay for PDF
                            st.session_state['gradcam_overlay'] = overlay_img
                        else:
                            st.warning("Could not generate Grad-CAM visualization")
                else:
                    st.warning("Grad-CAM not available for this model architecture")
                
                # Download report
                st.markdown("---")
                st.subheader("📄 Download Report")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prepare data for PDF
                    pdf_data = {
                        'disease': disease.replace('_', ' ').title(),
                        'crop': crop.title(),
                        'confidence': confidence,
                        'severity': disease_info['severity'],
                        'treatment': disease_info['treatment'],
                        'prevention': disease_info['prevention'],
                        'top_predictions': top_predictions
                    }
                    
                    pdf_buffer = generate_pdf_report(pdf_data)
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                with col2:
                    # CSV export
                    csv_data = pd.DataFrame([{
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Crop': crop.title(),
                        'Disease': disease.replace('_', ' ').title(),
                        'Confidence': f"{confidence:.2f}%",
                        'Severity': disease_info['severity']
                    }])
                    
                    csv_buffer = csv_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download CSV Data",
                        data=csv_buffer,
                        file_name=f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                with st.expander("🔍 Error Details"):
                    st.exception(e)
    
    # ========== TAB 2: YIELD PREDICTION ==========
    with tab2:
        st.header("📊 Crop Yield Prediction")
        st.markdown("Enter soil and climate parameters to get AI-powered crop recommendations")
        
        if yield_model is None:
            st.warning("⚠️ Yield prediction model not available")
        else:
            st.subheader("🌡️ Input Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Soil Nutrients**")
                N = st.number_input("Nitrogen (N)", value=90.0, min_value=0.0, max_value=200.0, step=1.0)
                P = st.number_input("Phosphorus (P)", value=42.0, min_value=0.0, max_value=150.0, step=1.0)
                K = st.number_input("Potassium (K)", value=43.0, min_value=0.0, max_value=200.0, step=1.0)
            
            with col2:
                st.markdown("**Climate Conditions**")
                temperature = st.number_input("Temperature (°C)", value=20.9, min_value=0.0, max_value=50.0, step=0.1)
                humidity = st.number_input("Humidity (%)", value=82.0, min_value=0.0, max_value=100.0, step=1.0)
                rainfall = st.number_input("Rainfall (mm)", value=202.9, min_value=0.0, max_value=500.0, step=1.0)
            
            with col3:
                st.markdown("**Soil Properties**")
                pH = st.number_input("Soil pH", value=6.5, min_value=0.0, max_value=14.0, step=0.1)
                st.markdown("&nbsp;")
                predict_button = st.button("🚀 Get Recommendation", type="primary", use_container_width=True)
            
            if predict_button:
                features = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
                
                try:
                    with st.spinner('🔄 Analyzing conditions...'):
                        if hasattr(yield_model, 'predict_proba'):
                            prediction = yield_model.predict(features)
                            probabilities = yield_model.predict_proba(features)[0]
                            
                            predicted_crop = prediction[0]
                            classes = yield_model.classes_
                            
                            st.success(f"### 🌾 Recommended Crop: **{predicted_crop}**")
                            
                            # Top 3 recommendations
                            top_3_idx = np.argsort(probabilities)[-3:][::-1]
                            
                            st.markdown("#### Top 3 Suitable Crops")
                            
                            for i, idx in enumerate(top_3_idx):
                                crop_name = classes[idx]
                                suitability = probabilities[idx] * 100
                                
                                if i == 0:
                                    st.success(f"🥇 **{crop_name}**: {suitability:.1f}% suitability")
                                elif i == 1:
                                    st.info(f"🥈 **{crop_name}**: {suitability:.1f}% suitability")
                                else:
                                    st.warning(f"🥉 **{crop_name}**: {suitability:.1f}% suitability")
                        else:
                            prediction = yield_model.predict(features)
                            st.success(f"### Predicted Yield: **{prediction[0]:.2f} tons/hectare**")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    # ========== TAB 3: USER GUIDE ==========
    with tab3:
        st.header("📖 User Guide")
        
        st.markdown("""
        ### 🎯 How to Use This System
        
        #### Disease Detection:
        1. **Prepare Your Image**
           - Use natural daylight for best results
           - Ensure the affected leaf is clearly visible
           - Avoid shadows and reflections
           - Minimum recommended resolution: 800x800 pixels
        
        2. **Upload & Analyze**
           - Click "Browse files" in the Disease Detection tab
           - Select your image (JPG, PNG supported)
           - Wait for AI analysis (usually 2-5 seconds)
        
        3. **Interpret Results**
           - **Green (>70%)**: High confidence - reliable diagnosis
           - **Yellow (50-70%)**: Moderate confidence - likely accurate
           - **Red (<50%)**: Low confidence - consider retaking photo
        
        4. **Take Action**
           - Review treatment recommendations
           - Implement prevention measures
           - Consult local agricultural extension if needed
        
        #### Yield Prediction:
        1. Input accurate soil and climate data
        2. Click "Get Recommendation"
        3. Review top 3 suitable crops for your conditions
        
        ### ⚠️ Important Notes
        - This is an AI assistant tool, not a replacement for professional diagnosis
        - For severe infestations, consult agricultural experts
        - Model accuracy: ~75% on validation data
        - Best results with clear, well-lit images
        
        ### 📞 Support
        For technical issues or questions, contact your agricultural extension service.
        
        ### 🔬 Model Information
        - **Architecture**: MobileNetV2 (optimized for efficiency)
        - **Training Dataset**: CCMT (25,000+ images)
        - **Supported Crops**: Cashew, Cassava, Maize, Tomato
        - **Disease Classes**: 22 (including healthy states)
        - **Validation Accuracy**: 75.5%
        - **Top-3 Accuracy**: 97.3%
        - **This Project was design and implemented by Mohammed Abdulrafiu Omotosho
        """)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')} | Version 2.0")

if __name__ == "__main__":
    main()
