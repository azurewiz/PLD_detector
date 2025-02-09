import os
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Set Page Configuration
st.set_page_config(page_title="Potato Leaf Disease Detector", page_icon="🥔", layout="wide")

# Load Model
model = load_model("models/potato_leaf_model.h5")
# Custom CSS Styling
st.markdown("""
    <style>
        .main { background-color: #F0F2F6; }
        .stButton>button { background-color: #FF4B4B; color: white; border-radius: 10px; font-size: 18px; }
        .stFileUploader { border-radius: 10px; }
        .stImage { border-radius: 10px; }
        .stTitle { text-align: center; font-size: 24px; font-weight: bold; }
        .prediction-box { padding: 20px; border-radius: 10px; background-color: #ffffff; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for Navigation
st.sidebar.image("assets/potato-chips-23994.png")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["📂 Upload Image", "📸 Use Webcam", "ℹ️ About"])

# Function to Predict Leaf Disease
def predict_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    classes = ["🍂 Early Blight", "🍁 Late Blight", "🍃 Healthy"]
    return classes[np.argmax(predictions)]

#  Upload Image Section
if page == "📂 Upload Image":
    st.title("🥔 Potato Leaf Disease Detection")
    st.write("Upload an image of a potato leaf, and the model will classify it.")
    
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        # Prediction with Loading Animation
        with st.spinner("🔍 Analyzing the leaf..."):
            result = predict_image(image)
        
        # Display Prediction Result
        st.markdown(f"<div class='prediction-box'><h2>🌟 Prediction: {result}</h2></div>", unsafe_allow_html=True)

#  Use Webcam Section
elif page == "📸 Use Webcam":
    st.title("📷 Live Potato Leaf Detection")
    st.write("Capture an image using your webcam.")

    cam = st.button("📸 Capture Image from Webcam")
    if cam:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(img, caption="📸 Captured Image", use_container_width=True)

            # Prediction with Loading Animation
            with st.spinner("🔍 Analyzing the leaf..."):
                result = predict_image(img)

            # Display Prediction Result
            st.markdown(f"<div class='prediction-box'><h2>🌟 Prediction: {result}</h2></div>", unsafe_allow_html=True)

#  About Section
else:
    st.title("ℹ️ About This Project")
    st.write("""
        - 🔬 **Developed using Deep Learning (CNN + TensorFlow)**
        - 🖥️ **Deployed with Streamlit for an Interactive UI**
        - 📸 **Supports Image Upload and Live Webcam Capture**
        - 🌱 **Provides Leaf Health Insights**
    """)

    st.markdown("**Developer: Sai Arvind Arun** ")
