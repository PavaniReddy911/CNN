# app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Replace with your trained model path
model = tf.keras.models.load_model("model/road_damage_model.h5")

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
# ABOUT PROJECT SECTION
# --------------------------------------------------

st.subheader("📘 About the Project")

st.write("""
Road monitoring is essential for preventing accidents, reducing maintenance costs,
and improving transportation safety.

This system uses Convolutional Neural Networks (CNNs) to automatically detect
road damages such as potholes, cracks, and manholes from road images.

### Industry Applications
- Smart City Infrastructure
- Government Road Monitoring
- Autonomous Vehicles
- Traffic Safety Systems
""")

st.divider()

# --------------------------------------------------
# IMAGE UPLOAD SECTION
# --------------------------------------------------

st.subheader("📂 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# IMAGE PREVIEW
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(image, use_container_width=True)

    # --------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------

    img = image.resize((224, 224))
    img_array = np.array(img)

    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]

    confidence = float(np.max(prediction) * 100)

    # --------------------------------------------------
    # SEVERITY LOGIC
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

    probs = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(classes, probs)

    ax.set_ylabel("Confidence (%)")
    ax.set_xlabel("Damage Type")
    ax.set_title("Class Confidence")

    st.pyplot(fig)

    st.divider()

    # --------------------------------------------------
    # RECOMMENDATION SECTION
    # --------------------------------------------------

    st.subheader("📌 Recommendations")

    if predicted_class == "Pothole":
        st.error("""
        Immediate maintenance recommended.
        High-risk road condition detected.
        """)

    elif predicted_class == "Crack":
        st.warning("""
        Schedule preventive repair.
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

st.caption("Developed using CNN + Deep Learning + Streamlit")
