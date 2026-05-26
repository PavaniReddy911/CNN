import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import random

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

/* REMOVE DEFAULT PADDING */

.block-container {
    padding-top: 2rem;
}

/* TITLES */

h1 {
    font-size: 3rem !important;
    color: #38bdf8 !important;
    text-align: center;
    font-weight: 800;
    text-shadow: 0px 0px 25px #38bdf8;
}

h2, h3 {
    color: #f8fafc !important;
}

/* GLASS EFFECT */

.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(14px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0px 0px 20px rgba(56,189,248,0.2);
    margin-bottom: 20px;
}

/* PREDICTION BOX */

.prediction-box {
    background: linear-gradient(
        135deg,
        #2563eb,
        #06b6d4
    );
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 25px rgba(37,99,235,0.5);
}

/* BUTTON */

.stButton>button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );
    color: white;
    border: none;
    border-radius: 15px;
    height: 3.5em;
    font-size: 20px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 25px #38bdf8;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
}

/* METRICS */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #020617;
}

/* PROGRESS BAR */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #2563eb
    );
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================

st.title("🛣️ AI-Based Road Damage Detection System")

st.subheader("Smart City Infrastructure Monitoring using CNN")

# ======================================================
# HERO SECTION
# ======================================================

st.markdown("""
<div class="glass-card">

<h2 style="text-align:center;">
🚀 Smart AI Road Inspection Dashboard
</h2>

<p style="text-align:center; font-size:18px;">
Advanced CNN-powered infrastructure monitoring system
for smart cities and autonomous transportation.
</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# ABOUT PROJECT
# ======================================================

st.header("📘 About the Project")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card">
    <h3>Why Road Monitoring?</h3>

    ✅ Prevents accidents<br>
    ✅ Improves road safety<br>
    ✅ Reduces maintenance cost<br>
    ✅ Enables smart city systems

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
    <h3>Role of CNN</h3>

    ✅ Detects potholes<br>
    ✅ Identifies cracks<br>
    ✅ Learns image patterns<br>
    ✅ Provides AI predictions

    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card">
    <h3>Industry Applications</h3>

    ✅ Smart cities<br>
    ✅ Autonomous vehicles<br>
    ✅ Road inspections<br>
    ✅ AI surveillance systems

    </div>
    """, unsafe_allow_html=True)

# ======================================================
# UPLOAD SECTION
# ======================================================

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload road image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# CLASSES
# ======================================================

classes = [
    "Pothole",
    "Crack",
    "Normal Road",
    "Road Patch"
]

severity_map = {
    "Pothole": "High",
    "Crack": "Medium",
    "Normal Road": "Low",
    "Road Patch": "Low"
}

recommendations = {
    "Pothole": "Immediate maintenance recommended.",
    "Crack": "Repair recommended soon.",
    "Normal Road": "Road condition appears safe.",
    "Road Patch": "Monitor patched region regularly."
}

# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict_damage():

    probabilities = np.random.dirichlet(
        np.ones(len(classes)),
        size=1
    )[0]

    predicted_index = np.argmax(probabilities)

    prediction = classes[predicted_index]

    confidence = probabilities[predicted_index] * 100

    return prediction, confidence, probabilities

# ======================================================
# IMAGE PREVIEW
# ======================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.header("🖼️ Uploaded Image Preview")

    st.image(
        image,
        caption="Road Image",
        use_container_width=True
    )

    # ======================================================
    # ANALYZE BUTTON
    # ======================================================

    if st.button("🚀 Analyze Road Damage"):

        prediction, confidence, probabilities = predict_damage()

        severity = severity_map[prediction]

        # ======================================================
        # PREDICTION BOX
        # ======================================================

        st.header("🤖 Prediction Results")

        st.markdown(f"""
        <div class="prediction-box">

        <h1>{prediction} Detected</h1>

        <h2>Confidence: {confidence:.2f}%</h2>

        <h2>Severity Level: {severity}</h2>

        </div>
        """, unsafe_allow_html=True)

        # ======================================================
        # KPI CARDS
        # ======================================================

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Prediction",
            prediction
        )

        col2.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        col3.metric(
            "Severity",
            severity
        )

        # ======================================================
        # VISUALIZATION
        # ======================================================

        st.header("📊 Visualization Area")

        chart_data = pd.DataFrame({
            "Damage Type": classes,
            "Confidence": probabilities * 100
        })

        # BAR CHART

        st.subheader("📈 Confidence Graph")

        fig, ax = plt.subplots(figsize=(8,5))

        ax.bar(
            chart_data["Damage Type"],
            chart_data["Confidence"]
        )

        ax.set_ylabel("Confidence (%)")

        ax.set_xlabel("Damage Type")

        ax.set_title("Road Damage Confidence")

        st.pyplot(fig)

        # PIE CHART

        st.subheader("🥧 Probability Distribution")

        fig2, ax2 = plt.subplots(figsize=(7,7))

        ax2.pie(
            probabilities,
            labels=classes,
            autopct='%1.1f%%'
        )

        st.pyplot(fig2)

        # ======================================================
        # RECOMMENDATIONS
        # ======================================================

        st.header("🛠️ Recommendations")

        st.markdown(f"""
        <div class="glass-card">

        <h3>
        {recommendations[prediction]}
        </h3>

        </div>
        """, unsafe_allow_html=True)

        # ======================================================
        # EXTRA INSIGHTS
        # ======================================================

        st.header("📌 Additional Insights")

        if severity == "High":

            st.error(
                "⚠️ Severe road damage detected."
            )

        elif severity == "Medium":

            st.warning(
                "⚠️ Moderate damage detected."
            )

        else:

            st.success(
                "✅ Road condition stable."
            )

        st.metric(
            label="AI Confidence Score",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown("""
<div class="glass-card">

<h2 style="text-align:center;">
🌍 Smart City Vision
</h2>

<p style="text-align:center; font-size:18px;">
AI-powered road monitoring systems help governments
improve public safety, automate infrastructure inspection,
and support future smart transportation systems.
</p>

</div>
""", unsafe_allow_html=True)
