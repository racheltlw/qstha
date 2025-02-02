import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


model_med = joblib.load("mediation_classifier.pkl")  
settle_model = joblib.load("settle_classifier.pkl")  
balance_classifier = joblib.load("balance_classifier.pkl")  

# feature structure based on training data
feature_columns = ['type_of_dispute_Commercial Entities/ And an Individual',
       'type_of_dispute_Family', 'type_of_dispute_Friends',
       'type_of_dispute_Neighbour', 'type_of_dispute_Others',
       'type_of_intake_Applications through Contact Centre',
       'type_of_intake_Court-Ordered', 'type_of_intake_Courts',
       'type_of_intake_Direct Intake', 'type_of_intake_External Agency',
       'type_of_intake_External Agency Referrals - Housing Development Board (HDB)',
       'type_of_intake_External Agency Referrals - Member of Parliament',
       'type_of_intake_External Agency Referrals - Singapore Police Force (SPF)',
       'type_of_intake_Others', 'type_of_intake_Walk Ins', 'year_registered',
       'month_registered', 'day_registered', 'quarter_registered']

# define dropdown options 
dispute_options = ["Neighbour", "Others", "Friends", "Family", "Commercial Entities/ And an Individual"]
intake_options = [
    "Courts",
    "External Agency",
    "Direct Intake",
    "Court-Ordered",
    "Applications through Contact Centre",
    "External Agency Referrals - Singapore Police Force (SPF)",
    "External Agency Referrals - Housing Development Board (HDB)",
    "External Agency Referrals - Member of Parliament",
    "Others",
    "Walk Ins"
]

def process_new_data(type_of_dispute, type_of_intake, date_registered, feature_columns):
    """
    Converts user inputs into a feature array matching the model's training data.
    """
    year_registered = date_registered.year
    month_registered = date_registered.month
    day_registered = date_registered.day
    quarter_registered = (month_registered - 1) // 3 + 1


    feature_values = {col: 0 for col in feature_columns}

    dispute_col_name = f"type_of_dispute_{type_of_dispute}"
    if dispute_col_name in feature_values:
        feature_values[dispute_col_name] = 1

    intake_col_name = f"type_of_intake_{type_of_intake}"
    if intake_col_name in feature_values:
        feature_values[intake_col_name] = 1

    feature_values["year_registered"] = year_registered
    feature_values["month_registered"] = month_registered
    feature_values["day_registered"] = day_registered
    feature_values["quarter_registered"] = quarter_registered

    input_array = np.array([feature_values[col] for col in feature_columns]).reshape(1, -1)
    return input_array

# Streamlit App Layout
st.title("Community Mediation Outcome Predictor")
st.write("Predict whether a case will be mediated and settled based on registration details.")

# User Inputs
date_registered = st.date_input("Select Registration Date", datetime.today())
type_of_dispute = st.selectbox("Select Type of Dispute", dispute_options)
type_of_intake = st.selectbox("Select Type of Intake", intake_options)

# Prediction Button
if st.button("Predict Mediation Outcome"):
    # Process new data
    new_input = process_new_data(type_of_dispute, type_of_intake, date_registered, feature_columns)


    med_prob = model_med.predict_proba(new_input)[:, 1][0]

    #threshold for prediction of mediation 
    if med_prob > 0.5:
        settle_prob = settle_model.predict_proba(new_input)[:, 1][0]
    else:
        settle_prob = 0


    stacked_features = np.array([[med_prob, settle_prob]])


    predicted_class = balance_classifier.predict(stacked_features)[0]

    prediction_map = {
        0: "Not Mediated",
        1: "Mediation Without Settlement",
        2: "Mediation With Settlement"
    }


    st.subheader("Prediction Result:")
    st.write(f"📌 The case is predicted to be: **{prediction_map[predicted_class]}**")

