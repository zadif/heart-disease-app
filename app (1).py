import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_svc_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    # must match the notebook exactly
    return {
        "Sex": LabelEncoder().fit(["Male", "Female"]),
        "ChestPain": LabelEncoder().fit(["Typical angina", "Atypical angina",
                                         "Non-anginal pain", "Asymptomatic"]),
        "ExerciseAngina": LabelEncoder().fit(["Yes", "No"]),
        "STSlope": LabelEncoder().fit(["Upsloping", "Flat", "Downsloping"]),
    }

model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    "Enter the patient's details and the trained Support Vector Classifier (SVC) "
    "will predict whether heart disease is present. "
    "*Built on the UCI Heart Disease (Cleveland) dataset using 4 input attributes.*"
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type",
                      ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
with col2:
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    slope = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [cp],
        "ExerciseAngina": [exang],
        "STSlope": [slope],
    })
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col])
                            for col in user_input.columns})

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE PREDICTED")
    else:
        st.success("### ✅ NO HEART DISEASE PREDICTED")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Educational demo only. This is not a medical diagnosis tool.")