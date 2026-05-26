# app.py

import streamlit as st
from PIL import Image
import numpy as np
import joblib
import cv2

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Road Damage Detection",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

# Load trained ML model
model = joblib.load("model/road_damage_model.pkl")

# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------

classes = ["Pothole", "Crack", "Manhole"]

# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------

st.markdown("""
# 🚧 AI-Based Road Damage Detection System
### Smart City Infrastructure Monitoring using CNN
""")

st.divider()

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.subheader("📘 About the Project")

st.write("""
Road monitoring is important for reducing accidents
and improving transportation safety.

This AI system detects road damages like potholes,
cracks, and manholes using image analysis.

### Industry Applications
- Smart Cities
- Road Safety Monitoring
- Government Infrastructure
- Autonomous Vehicles
""")

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📂 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------

    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(image, use_container_width=True)

    # --------------------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------------------

    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (128, 128))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Flatten image
    flat_img = gray.flatten()

    # Normalize
    flat_img = flat_img / 255.0

    # Reshape for model
    flat_img = flat_img.reshape(1, -1)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = model.predict(flat_img)[0]

    probabilities = model.predict_proba(flat_img)[0]

    confidence = np.max(probabilities) * 100

    predicted_class = classes[prediction]

    # --------------------------------------------------
    # SEVERITY LEVEL
    # --------------------------------------------------

    if confidence < 40:
        severity = "Low"

    elif confidence < 75:
        severity = "Medium"

    else:
        severity = "High"

    # --------------------------------------------------
    # PREDICTION AREA
    # --------------------------------------------------

    with col2:

        st.subheader("🤖 Prediction Results")

        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence: {confidence:.2f}%")

        st.warning(f"Severity: {severity}")

    st.divider()

    # --------------------------------------------------
    # VISUALIZATION AREA
    # --------------------------------------------------

    st.subheader("📊 Prediction Visualization")

    probs = probabilities * 100

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(classes, probs)

    ax.set_xlabel("Damage Type")

    ax.set_ylabel("Confidence (%)")

    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------

    st.subheader("📌 Recommendations")

    if predicted_class == "Pothole":

        st.error("""
        Immediate maintenance recommended.
        High-risk road condition detected.
        """)

    elif predicted_class == "Crack":

        st.warning("""
        Preventive repair suggested.
        Moderate road damage identified.
        """)

    elif predicted_class == "Manhole":

        st.info("""
        Inspection required for public safety.
        Possible infrastructure issue detected.
        """)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption("Developed using Machine Learning + Streamlit")
