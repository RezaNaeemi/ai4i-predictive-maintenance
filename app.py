
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------
# 1. Page Configuration
# --------------------------------



st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Predictive Maintenance")
st.write(
    "Enter machine operating conditions to estimate "
    "the probability of machine failure."
)

# --------------------------------
# 2. Load Model and Threshold
# --------------------------------


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.joblib"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold

try:
    model, threshold = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Please check the models folder."
    )
    st.stop()

# --------------------------------
# 3. User Input
# --------------------------------

st.subheader("Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0,
        max_value=5000,
        value=1500,
        step=10
    )

with col2:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=500,
        value=100,
        step=1
    )

# --------------------------------
# 4. Prediction
# --------------------------------

if st.button("Predict Machine Failure", type="primary"):

    # Check input consistency
    if process_temperature < air_temperature:
        st.error(
            "Process temperature should not be lower "
            "than air temperature."
        )
        st.stop()

    if rotational_speed == 0:
        st.error("Rotational speed must be greater than zero.")
        st.stop()

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Feature Engineering
    input_data["Temperature Difference [K]"] = (
        input_data["Process temperature [K]"]
        - input_data["Air temperature [K]"]
    )

    input_data["Power [W]"] = (
        input_data["Torque [Nm]"]
        * input_data["Rotational speed [rpm]"]
        * (2 * np.pi / 60)
    )

    input_data["Wear Torque Interaction"] = (
        input_data["Tool wear [min]"]
        * input_data["Torque [Nm]"]
    )

    # Predict probability
    failure_probability = model.predict_proba(
        input_data
    )[0, 1]

    # Apply selected threshold
    prediction = int(failure_probability >= threshold)

    # --------------------------------
    # 5. Display Results
    # --------------------------------

    st.subheader("Prediction Results")

    st.metric(
        "Failure Probability",
        f"{failure_probability * 100:.2f}%"
    )

    st.progress(float(failure_probability))

    if prediction == 1:
        st.error(
            "⚠️ Potential Machine Failure Detected"
        )
        st.write(
            "The predicted failure probability is above "
            "the selected classification threshold."
        )
    else:
        st.success(
            "✅ No Failure Predicted"
        )
        st.write(
            "The predicted failure probability is below "
            "the selected classification threshold."
        )

    st.caption(
        f"Classification threshold: {threshold:.4f}"
    )

    st.info(
        "This is a machine-learning prediction, not a "
        "guarantee of actual machine condition."
    )
