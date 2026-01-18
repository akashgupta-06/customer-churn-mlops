# app/app.py

import os
import sys

# Add project root to Python path so `src` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
from src.predict import predict_single


st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Customer Churn Prediction System")
st.write(
    "Enter customer details to predict churn risk using the production ML model."
)

st.divider()

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, value=5)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_sec = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device = st.selectbox("Device Protection", ["Yes", "No"])
        tech = st.selectbox("Tech Support", ["Yes", "No"])
        tv = st.selectbox("Streaming TV", ["Yes", "No"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    monthly = st.number_input("Monthly Charges", min_value=0.0, value=85.5)
    total = st.number_input("Total Charges", min_value=0.0, value=430.0)

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    try:
        result = predict_single(customer)

        st.subheader("Prediction Result")

        prob = result["churn_probability"]
        risk = result["risk_level"]

        if risk == "High":
            st.error(f"High Churn Risk\n\nProbability: {prob}")
        else:
            st.success(f"Low Churn Risk\n\nProbability: {prob}")

        st.write("Detailed Output:")
        st.json(result)

    except Exception as e:
        st.exception(e)
