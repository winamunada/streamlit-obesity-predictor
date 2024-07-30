import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load('obesity_model.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title('Obesity Prediction Web App')

# User inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
gender = st.selectbox('Gender', ['Male', 'Female'])
height = st.number_input('Height (in meters)', min_value=0.5, max_value=2.5, value=1.75)
weight = st.number_input('Weight (in kg)', min_value=10, max_value=300, value=70)
calories = st.selectbox('Daily caloric intake frequency', ['no', 'Sometimes', 'Frequently'])
favc = st.selectbox('Frequent consumption of high-caloric food', ['no', 'yes'])
fcvc = st.number_input('Frequency of vegetable consumption', min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input('Number of main meals', min_value=1.0, max_value=6.0, value=3.0)
scc = st.selectbox('Consumption of food between meals', ['no', 'yes'])
smoke = st.selectbox('Smokes', ['no', 'yes'])
ch2o = st.number_input('Daily water intake (liters)', min_value=1.0, max_value=3.0, value=2.0)
family_history_with_overweight = st.selectbox('Family history with overweight', ['no', 'yes'])
faf = st.number_input('Physical activity frequency (days per week)', min_value=0.0, max_value=7.0, value=2.0)
tue = st.number_input('Time spent on technology (hours per day)', min_value=0.0, max_value=24.0, value=3.0)
caec = st.selectbox('Consumption of alcohol', ['no', 'Sometimes', 'Frequently'])
mtrans = st.selectbox('Transportation method', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

# Create the input data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'CALC_Sometimes': [1 if calories == 'Sometimes' else 0],
    'CALC_Frequently': [1 if calories == 'Frequently' else 0],
    'FAVC_yes': [1 if favc == 'yes' else 0],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'SCC_yes': [1 if scc == 'yes' else 0],
    'SMOKE_yes': [1 if smoke == 'yes' else 0],
    'CH2O': [ch2o],
    'family_history_with_overweight_yes': [1 if family_history_with_overweight == 'yes' else 0],
    'FAF': [faf],
    'TUE': [tue],
    'CAEC_Sometimes': [1 if caec == 'Sometimes' else 0],
    'CAEC_Frequently': [1 if caec == 'Frequently' else 0],
    'MTRANS_Motorbike': [1 if mtrans == 'Motorbike' else 0],
    'MTRANS_Bike': [1 if mtrans == 'Bike' else 0],
    'MTRANS_Public_Transportation': [1 if mtrans == 'Public_Transportation' else 0],
    'MTRANS_Walking': [1 if mtrans == 'Walking' else 0],
    'Gender_Female': [1 if gender == 'Female' else 0]
})

# Ensure the input data has the same columns as the training data
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Predict obesity category
prediction = model.predict(input_data)[0]

st.write(f'The predicted obesity category is: {prediction}')
