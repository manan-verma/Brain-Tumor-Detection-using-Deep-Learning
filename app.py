import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page config for a widescreen layout and dark theme setup
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme, glassmorphism cards, custom fonts, and glowing elements
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main body typography & color scheme */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #f0f2f6;
    }
    
    .stApp {
        background-color: #0b0f19;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(23, 28, 42, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .glow-card {
        background: rgba(23, 28, 42, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.08);
    }

    /* Header styling with gradient */
    .title-gradient {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #a0aabf;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }

    /* Custom labels and status */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        border: 1px solid rgba(26, 188, 156, 0.3);
        margin-bottom: 15px;
    }
    
    /* Progress bar styles */
    .prob-label {
        font-weight: 600;
        margin-bottom: 4px;
        font-size: 0.95rem;
        display: flex;
        justify-content: space-between;
    }
    
    .prob-val {
        color: #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Helper Functions & Caching -----------------

# Load model and cache it to avoid reloading on every run
@st.cache_resource
def load_tumor_model():
    model_path = "model.h5"
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Error loading model from file: {e}")
            return None
    return None

# Perform prediction and image preprocessing
def predict_tumor(image, model):
    # Resize image to match training target size
    img = image.convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)[0]
    return preds

# Map labels as constructed in standard os.listdir(train_dir)
CLASSES = ["Glioma", "Meningioma", "No Tumor Detected", "Pituitary"]

CLINICAL_INFO = {
    "Glioma": {
        "description": "Gliomas are primary brain tumors that start in glial cells, which surround and support nerve cells. They represent the most common type of malignant brain tumor.",
        "symptoms": "Headaches, nausea, seizures, difficulty speaking, cognitive decline, or physical weakness on one side of the body.",
        "urgency": "High. Gliomas require urgent neurosurgical evaluation, imaging tests (like contrast MRI), and potentially a biopsy to determine the exact grade and treatment plan."
    },
    "Meningioma": {
        "description": "Meningiomas arise from the meninges—the protective membranes covering the brain and spinal cord. The majority are slow-growing and benign (Grade I).",
        "symptoms": "Vision changes, worsening headaches, hearing loss, loss of smell, or localized weaknesses.",
        "urgency": "Moderate. While often benign, they can press on brain tissue. Consultation with a neurosurgeon is recommended to evaluate if active surveillance, surgery, or radiation is needed."
    },
    "No Tumor Detected": {
        "description": "The AI model has analyzed the scan and found no features matching glioma, meningioma, or pituitary tumors.",
        "symptoms": "If the patient is experiencing headaches, dizziness, or neurological deficits, other non-tumor causes should be investigated.",
        "urgency": "Low/Routine. Discuss the results with the ordering physician for general checkups and diagnostic alternatives."
    },
    "Pituitary": {
        "description": "Pituitary tumors develop in the pituitary gland at the base of the brain. They are typically benign adenomas but can secrete excessive hormones or cause visual issues by pressing on the optic chiasm.",
        "symptoms": "Hormonal imbalances (unexplained weight changes, fatigue), vision loss (especially peripheral), or headaches.",
        "urgency": "Moderate. Management often involves neurosurgery or endocrinology to control hormone levels or decompress nerves."
    }
}

# ----------------- App Layout & Structure -----------------

model = load_tumor_model()

# Header Section
st.markdown('<h1 class="title-gradient">🧠 Brain Tumor Detection AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Deep Learning Platform for Brain Tumor Classification</p>', unsafe_allow_html=True)

# Sidebar setup
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("System Status")
    if model is not None:
        st.markdown('<span class="status-badge">● Model Loaded</span>', unsafe_allow_html=True)
        st.write("**Model Architecture:** VGG16 + Dense Classifier")
        st.write("**Input Size:** 128 x 128 px (RGB)")
    else:
        st.error("❌ model.h5 not found in workspace")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("App Instructions")
    st.write("1. Upload a brain MRI scan (T1/T2 contrast preferred).")
    st.write("2. Alternatively, click one of the quick-load sample images.")
    st.write("3. Review predicted categories, probabilities, and clinical next steps.")
    st.markdown('</div>', unsafe_allow_html=True)

# Main Section
col_upload, col_result = st.columns([1, 1.2], gap="large")

# Selected image state management
if "selected_img_path" not in st.session_state:
    st.session_state.selected_img_path = None

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("1. Input Brain MRI Scan")
    
    # Image file uploader
    uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])
    
    # Handle sample selector
    st.markdown("<p style='margin-top: 15px; font-weight: 600; color: #a0aabf;'>Or select a preloaded sample MRI:</p>", unsafe_allow_html=True)
    
    samples = {
        "Glioma Sample": "Te-gl_0015.jpg",
        "Meningioma Sample": "Te-meTr_0001.jpg",
        "No Tumor Sample": "Te-noTr_0004.jpg",
        "Pituitary Sample": "Te-piTr_0003.jpg"
    }
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Glioma MRI", use_container_width=True):
            st.session_state.selected_img_path = samples["Glioma Sample"]
        if st.button("No Tumor MRI", use_container_width=True):
            st.session_state.selected_img_path = samples["No Tumor Sample"]
    with col_s2:
        if st.button("Meningioma MRI", use_container_width=True):
            st.session_state.selected_img_path = samples["Meningioma Sample"]
        if st.button("Pituitary MRI", use_container_width=True):
            st.session_state.selected_img_path = samples["Pituitary Sample"]
            
    # Reset selected path if a custom file is uploaded
    if uploaded_file is not None:
        st.session_state.selected_img_path = None

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load and display selected image
    selected_image = None
    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
    elif st.session_state.selected_img_path is not None and os.path.exists(st.session_state.selected_img_path):
        selected_image = Image.open(st.session_state.selected_img_path)
        st.caption(f"Loaded Sample: `{st.session_state.selected_img_path}`")
        
    if selected_image is not None:
        st.markdown('<div class="glass-card" style="text-align: center;">', unsafe_allow_html=True)
        st.image(selected_image, caption="Active MRI Scan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_result:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("2. Diagnostics & Diagnostics breakdown")
    
    if model is None:
        st.warning("⚠️ Model is not loaded. Please make sure model.h5 is in the project root directory.")
    elif selected_image is None:
        st.info("💡 Please upload an MRI scan or select a sample scan on the left to begin diagnostics.")
    else:
        with st.spinner("Processing MRI through neural network layers..."):
            # Run predictions
            predictions = predict_tumor(selected_image, model)
            pred_class_idx = np.argmax(predictions)
            pred_class_name = CLASSES[pred_class_idx]
            confidence = predictions[pred_class_idx] * 100
            
            # Prediction Card
            st.markdown(f"""
            <div class="glow-card">
                <h4 style="margin-top:0; color:#00f2fe; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px;">AI Diagnostics Result</h4>
                <h2 style="margin: 10px 0; color:#ffffff; font-size: 2.2em; font-weight:700;">{pred_class_name}</h2>
                <div style="font-size: 1.1em; color: #a0aabf;">Confidence Score: <strong style="color: #00f2fe; font-size: 1.25em;">{confidence:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability Breakdown Dashboard
            st.markdown("##### Probability Breakdown")
            for idx, label in enumerate(CLASSES):
                prob = predictions[idx]
                st.markdown(f"""
                <div class="prob-label">
                    <span>{label}</span>
                    <span class="prob-val">{prob*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob))
                
            # Clinical Education & Next Steps
            st.markdown("---")
            st.markdown("##### Clinical Overview")
            info = CLINICAL_INFO[pred_class_name]
            
            st.write(info["description"])
            
            st.markdown(f"**Common Clinical Symptoms:** {info['symptoms']}")
            
            st.markdown(f"**Recommended Urgency Level / Next Action:** {info['urgency']}")
            
            # Medical Disclaimer
            st.markdown("""
            <div style="margin-top: 25px; padding: 15px; border-left: 4px solid #f39c12; background: rgba(243, 156, 18, 0.08); border-radius: 4px; font-size: 0.85em; color: #e67e22;">
                <strong>⚠️ Clinical Disclaimer:</strong> This interface is an AI-assisted tool trained on dataset classes. It does not replace a clinical consultation or professional medical imaging interpretation. Patients must consult a qualified doctor or neurosurgeon for official medical diagnoses.
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
