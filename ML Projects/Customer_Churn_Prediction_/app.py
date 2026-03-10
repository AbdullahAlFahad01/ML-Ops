import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scal.pkl")

st.image("https://softwareequity.com/wp-content/uploads/2023/07/SEG_BlogHeader_CustomerChurnAnalysis.png")

st.markdown("# Customer Churn Prediction using ML")
st.divider()

st.write("Enter the customer details below to find the prediction")
st.divider()

# Inputs
age = st.number_input("Enter Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Select your gender", ["Male", "Female"])
tenure = st.number_input("Enter Tenure (years)", min_value=0, max_value=20, step=1)
ct = st.selectbox("Contract Type", ["One-Year", "Two-Year", "Month-To-Month"])
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# Button
if st.button("Predict"):
    # Encode categorical variables
    gender_selection = 1 if gender == "Female" else 0
    ct_selection = 1 if ct == "One-Year" else 2 if ct == "Two-Year" else 0

    # Prepare feature array
    X = np.array([[age, gender_selection, tenure, ct_selection, total_charges]])

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]
    predicted_label = "Churn" if prediction == 1 else "Not Churn"

    # Display result
    st.write(f"Prediction: **{predicted_label}**")