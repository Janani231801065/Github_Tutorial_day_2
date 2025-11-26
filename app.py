import streamlit as st
import cv2
import numpy as np
from cnn_numpy import convolve, relu, max_pooling, fully_connected, softmax, labels, weights, bias, edge_kernel

# ---------- Page Configuration ----------
st.set_page_config(page_title="🧠 CNN Image Classifier", layout="centered")

# ---------- Custom CSS Styling ----------
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #ede9fe, #c4b5fd, #a78bfa);
        color: #2e1065;
    }

    /* Title and headers */
    h1, h2, h3, h4 {
        color: #581c87;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Buttons */
    .stButton>button {
        background-color: #7e22ce;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 1rem;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #9333ea;
        color: #fff;
        transform: scale(1.05);
    }

    /* File uploader */
    .stFileUploader>div>div>button {
        background-color: #7e22ce !important;
        color: white !important;
        border-radius: 8px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #a78bfa, #c084fc, #d8b4fe);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- App Header ----------
st.title("NumPy CNN Image Classifier")
st.markdown("Upload a grayscale image to test the trained CNN model")

# ---------- Image Upload ----------
uploaded_file = st.file_uploader("📁 Upload an image (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Display image
    st.image(img, caption="🖼 Uploaded Image", use_container_width=True, channels="GRAY")

    # Preprocess
    img_resized = cv2.resize(img, (32, 32))
    img_resized = img_resized / 255.0

    # Forward pass
    conv_output = convolve(img_resized, edge_kernel)
    relu_output = relu(conv_output)
    pooled_output = max_pooling(relu_output)
    flat = pooled_output.flatten()
    output = fully_connected(flat, weights, bias)
    probs = softmax(output)

    # Prediction
    pred_idx = np.argmax(probs)
    pred_label = labels[pred_idx]

    # Display result
    st.markdown(f"<h3 style='color:#581c87;'>✅ Predicted ASL Letter: <b>{pred_label}</b></h3>", unsafe_allow_html=True)
    st.bar_chart(probs)

else:
    st.info("👆 Upload an image to see prediction results.")
