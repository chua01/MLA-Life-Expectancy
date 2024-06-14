import streamlit as st
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv('led.csv')

# lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')
encoder = joblib.load('encoder.pkl')  

# Streamlit app interface
st.title('Life Expectancy Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    country = st.sidebar.selectbox('Country', df['Country'].unique())
    year = st.sidebar.slider('Year', int(df['Year'].min()), int(df['Year'].max()), 2015)
    adult_mortality = st.sidebar.number_input('Adult Mortality', min_value=0, max_value=1000, value=263)
    income_composition_of_resources = st.sidebar.number_input('Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.479)
    schooling = st.sidebar.number_input('Schooling', min_value=0.0, max_value=20.0, value=10.1)
    hiv_aids = st.sidebar.number_input('HIV/AIDS', min_value=0.0, max_value=100.0, value=0.1)
    total_expenditure = st.sidebar.number_input('Total Expenditure', min_value=0.0, max_value=100.0, value=8.16)
    population = st.sidebar.number_input('Population', min_value=0, max_value=int(df['Population'].max()), value=33736494)
    bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=100.0, value=19.1)
    measles = st.sidebar.number_input('Measles', min_value=0, max_value=int(df['Measles'].max()), value=1154)
    
    data = {
        'Country': country,
        'Year': year,
        'AdultMortality': adult_mortality,
        'Incomecompositionofresources': income_composition_of_resources,
        'Schooling': schooling,
        'HIV/AIDS': hiv_aids,
        'Totalexpenditure': total_expenditure,
        'Population': population,
        'BMI': bmi,
        'Measles': measles
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Create a copy of the input data for display purposes
display_input_df = input_df.copy()

# Encode the input data
categorical_features = ['Country']
# input_data[categorical_features] = encoder.transform(input_data[categorical_features])
input_df[categorical_features] = encoder.transform(input_df[categorical_features])

# Predict the life expectancy
prediction = rf_model.predict(input_df)

st.subheader('User Input Parameters')
st.write(display_input_df)

st.subheader('Prediction')
st.write(f"Predicted Life Expectancy: {prediction[0]:.2f}")
