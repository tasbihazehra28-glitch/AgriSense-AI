import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "crop_model.joblib"
DATA_PATH = BASE_DIR / "data" / "crop_demo_data.csv"

st.set_page_config(page_title="AgriSense AI", page_icon="🌱", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🌱 AgriSense AI")
st.subheader("Smart Crop Recommendation & Farming Assistant")
st.write(
    "Enter basic soil and climate conditions to receive a crop suggestion from a "
    "machine-learning model. This is an educational prototype, not a substitute for "
    "local agricultural advice or soil testing."
)

with st.sidebar:
    st.header("About this project")
    st.write("**AI method:** Random Forest classifier")
    st.write("**Inputs:** N, P, K, temperature, humidity, pH, rainfall")
    st.write("**Output:** Suggested crop + model confidence")
    st.caption("The included dataset is synthetic demo data created for this prototype.")

left, right = st.columns([1, 1])

with left:
    st.header("Your field conditions")
    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=60.0, step=1.0)
    with col2:
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=40.0, step=1.0)
    with col3:
        potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=210.0, value=40.0, step=1.0)

    col4, col5 = st.columns(2)
    with col4:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=55.0, value=26.0, step=0.5)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with col5:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

    predict = st.button("Recommend a crop", type="primary", use_container_width=True)

with right:
    st.header("AI recommendation")
    if predict:
        features = pd.DataFrame([{
            "N": nitrogen, "P": phosphorus, "K": potassium,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall
        }])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities.max()) * 100

        st.success(f"Suggested crop: **{prediction.title()}**")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.progress(min(max(confidence / 100, 0.0), 1.0))
        st.warning(
            "Confidence is the model's score for this demo dataset; it is not a guarantee "
            "of yield or suitability in your exact location."
        )

        if ph < 5.5:
            st.info("Soil pH is acidic. Consider a local soil test before making amendments.")
        elif ph > 7.8:
            st.info("Soil pH is alkaline. Check local agronomy guidance for this crop.")
        else:
            st.info("Your entered pH is within a broad range commonly suitable for many crops; "
                    "confirm the crop-specific range locally.")

        st.markdown("**Before planting:** verify the recommendation with a local agriculture "
                    "extension officer, current weather, irrigation availability, and a real soil test.")
    else:
        st.info("Enter your field conditions and select **Recommend a crop** to see the model result.")
        st.image(
            "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80",
            caption="Healthy fields depend on local conditions and informed decisions.",
            use_container_width=True
        )

st.divider()
st.header("How it works")
st.markdown(
    "1. The app collects seven soil and climate measurements.  \n"
    "2. A trained Random Forest model compares the measurements with example crop patterns.  \n"
    "3. The model returns its most likely crop and a confidence score.  \n"
    "4. A farmer should validate the suggestion with local expertise."
)

with st.expander("View the demo dataset"):
    st.dataframe(pd.read_csv(DATA_PATH), use_container_width=True)
