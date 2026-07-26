import streamlit as st
import requests
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI Medical Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

BACKEND_URL = "https://advanced-ai-medical-intelligence-platform-unwx.onrender.com"

st.title("🧠 Advanced AI Medical Intelligence Platform")
st.write("Upload a Brain MRI image and click **Analyze MRI**.")

st.divider()

# -------------------------------
# Upload Image
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose a Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.subheader("Original MRI")
    st.image(image, width=300)

    if st.button("Analyze MRI"):

        with st.spinner("Analyzing MRI..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict",
                    files=files
                )

                if response.status_code != 200:
                    st.error("Prediction failed.")
                    st.stop()

                result = response.json()

            except Exception as e:
                st.error(f"Unable to connect to backend.\n\n{e}")
                st.stop()

        st.success("Analysis Completed")

        st.divider()

        # -------------------------------
        # Prediction
        # -------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", result["prediction"].title())

        with col2:
            st.metric("Confidence", f"{result['confidence']:.2f}%")

        st.divider()

        # -------------------------------
        # Probability Scores
        # -------------------------------
        st.subheader("Prediction Probabilities")

        for disease, probability in result["probabilities"].items():

            st.write(f"**{disease.title()}**")

            st.progress(float(probability) / 100)

            st.write(f"{probability:.2f}%")

        st.divider()

        # -------------------------------
        # Heatmap
        # -------------------------------
        st.subheader("Grad-CAM Heatmap")

        heatmap_url = BACKEND_URL + result["heatmap"]

        st.image(
            heatmap_url,
            width=300,
            caption="Grad-CAM Visualization"
        )

        st.divider()

        # -------------------------------
        # AI Medical Report
        # -------------------------------
        st.subheader("AI Medical Report")

        report = result["medical_report"]

        st.markdown("### Diagnosis Summary")
        st.write(report.get("diagnosis_summary", ""))

        st.markdown("### Clinical Explanation")
        st.write(report.get("clinical_explanation", ""))

        st.markdown("### Confidence Interpretation")
        st.write(report.get("confidence_interpretation", ""))

        st.markdown("### Recommended Next Steps")
        st.write(report.get("recommended_next_steps", ""))

        st.markdown("### Medical Disclaimer")
        st.info(report.get("medical_disclaimer", ""))