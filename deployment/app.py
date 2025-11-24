import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# --- Mappings for Label Encoding (based on alphabetical sort of unique values from original data) ---
TYPEOFCONTACT_MAP = {'Company Invited': 0, 'Self Inquiry': 1}
OCCUPATION_MAP = {'Free Lancer': 0, 'Large Business': 1, 'Salaried': 2, 'Small Business': 3}
GENDER_MAP = {'Fe Male': 0, 'Female': 1, 'Male': 2, 'Unaware': 3}
PRODUCTPITCHED_MAP = {'Basic': 0, 'Deluxe': 1, 'King': 2, 'Standard': 3, 'Super Deluxe': 4}
MARITALSTATUS_MAP = {'Divorced': 0, 'Married': 1, 'Single': 2}
DESIGNATION_MAP = {'AVP': 0, 'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'VP': 4}

# Download and load the model
MODEL_REPO_ID = "deepakpathania/tourism-xgboost-model"
MODEL_FILENAME = "xgboost_model/best_tourism_model_v1.joblib"

try:
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model from Hugging Face Hub: {e}")
    st.stop()

# Streamlit UI for Wellness Tourism Package Purchase Prediction
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package.
Fill in the customer details below to get a prediction.
""")

# User input fields
st.header("Customer Details")

age = st.slider("Age", min_value=18, max_value=80, value=35)
type_of_contact = st.selectbox("Type of Contact", list(TYPEOFCONTACT_MAP.keys()))
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.slider("Duration of Pitch (minutes)", min_value=5, max_value=100, value=15)
occupation = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))
gender = st.selectbox("Gender", list(GENDER_MAP.keys())) # Using full list due to EDA observation
number_of_person_visiting = st.slider("Number of Persons Visiting", min_value=1, max_value=5, value=3)
number_of_followups = st.slider("Number of Follow-ups", min_value=1, max_value=6, value=3)
product_pitched = st.selectbox("Product Pitched", list(PRODUCTPITCHED_MAP.keys()))
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", list(MARITALSTATUS_MAP.keys()))
number_of_trips = st.slider("Number of Trips Annually", min_value=1, max_value=25, value=3)
passport = st.selectbox("Passport Holder?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
number_of_children_visiting = st.slider("Number of Children Visiting (under 5)", min_value=0, max_value=3, value=1)
designation = st.selectbox("Designation", list(DESIGNATION_MAP.keys()))
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0)

# Prepare input data for the model (matching Xtrain structure after LabelEncoding)
if st.button("Predict Purchase"):
    # Convert categorical inputs to numerical using defined mappings
    encoded_type_of_contact = TYPEOFCONTACT_MAP[type_of_contact]
    encoded_occupation = OCCUPATION_MAP[occupation]
    encoded_gender = GENDER_MAP[gender]
    encoded_product_pitched = PRODUCTPITCHED_MAP[product_pitched]
    encoded_marital_status = MARITALSTATUS_MAP[marital_status]
    encoded_designation = DESIGNATION_MAP[designation]

    # Create a DataFrame with the same column order as Xtrain
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': encoded_type_of_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': encoded_occupation,
        'Gender': encoded_gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'NumberOfFollowups': number_of_followups,
        'ProductPitched': encoded_product_pitched,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': encoded_marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'Designation': encoded_designation,
        'MonthlyIncome': monthly_income
    }])

    # Ensure column order matches Xtrain used during training
    # This list should match the column order in Xtrain exactly.
    # Using a predefined list as inference from Xtrain.columns from kernel state is reliable.
    column_order = [
        'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender',
        'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
        'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
        'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'Designation',
        'MonthlyIncome'
    ]
    input_data = input_data[column_order]

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[:, 1][0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **Customer WILL purchase the Wellness Tourism Package!** (Probability: {prediction_proba:.2f})")
    else:
        st.info(f"The model predicts: **Customer will NOT purchase the Wellness Tourism Package.** (Probability: {prediction_proba:.2f})")

    st.write("Note: The model's classification threshold is 0.45.")
