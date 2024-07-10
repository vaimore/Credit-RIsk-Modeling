import streamlit as st
import joblib
import pandas as pd
import os

# Load the saved XGBoost model using a relative path
model_path = os.path.join(os.path.dirname(__file__), 'best_xgb_model_smote.pkl')
model = joblib.load(model_path)

# Function to make predictions
def predict_default(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    return prediction[0], probability[0]

# Streamlit app
st.title("Credit Risk Prediction App")

# User inputs (example fields)
person_age = st.number_input("Enter person's age:", min_value=18, max_value=100, value=25)
person_income = st.number_input("Enter person's income:", min_value=0.0, value=50000.0)
loan_amnt = st.number_input("Enter loan amount:", min_value=0.0, value=10000.0)
loan_int_rate = st.number_input("Enter loan interest rate:", min_value=0.0, max_value=100.0, value=5.0)
loan_percent_income = st.number_input("Enter loan percent of income:", min_value=0.0, max_value=1.0, value=0.2)
person_emp_length = st.number_input("Enter person's employment length in years:", min_value=0.0, value=5.0)
cb_person_cred_hist_length = st.number_input("Enter credit history length:", min_value=0, max_value=50, value=10)
person_home_ownership_RENT = st.selectbox("Is the person renting?", [0, 1])
loan_grade_D = st.selectbox("Is the loan grade D?", [0, 1])
LGD = st.number_input("Enter Loss Given Default (LGD):", min_value=0.0, max_value=1.0, value=0.4)

# Collecting user inputs in a dictionary
user_input = {
    'person_age': person_age,
    'person_income': person_income,
    'person_emp_length': person_emp_length,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'person_home_ownership_MORTGAGE': 0,
    'person_home_ownership_OTHER': 0,
    'person_home_ownership_OWN': 0,
    'person_home_ownership_RENT': person_home_ownership_RENT,
    'loan_intent_DEBTCONSOLIDATION': 0,
    'loan_intent_EDUCATION': 0,
    'loan_intent_HOMEIMPROVEMENT': 0,
    'loan_intent_MEDICAL': 0,
    'loan_intent_PERSONAL': 0,
    'loan_intent_VENTURE': 0,
    'loan_grade_A': 0,
    'loan_grade_B': 0,
    'loan_grade_C': 0,
    'loan_grade_D': loan_grade_D,
    'loan_grade_E': 0,
    'loan_grade_F': 0,
    'loan_grade_G': 0,
    'cb_person_default_on_file_N': 0,
    'cb_person_default_on_file_Y': 0,
}

# Button to make prediction
if st.button("Predict"):
    prediction, probability = predict_default(user_input)
    st.write(f"Default Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Probability of Default: {probability:.2f}")

    # Calculate Exposure at Default (EAD) and Expected Loss (EL)
    EAD = loan_amnt
    EL = probability * EAD * LGD
    st.write(f"Exposure at Default (EAD): {EAD}")
    st.write(f"Expected Loss (EL): {EL:,2f}")
