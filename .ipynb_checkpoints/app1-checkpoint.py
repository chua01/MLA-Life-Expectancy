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
# encoder = joblib.load('encoder.pkl')  

# Streamlit app interface
st.title('Life Expectancy Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    # country = st.sidebar.selectbox('Country', df['Country'].unique())
    # year = st.sidebar.slider('Year', int(df['Year'].min()), int(df['Year'].max()), 2015)
    adult_mortality = st.sidebar.number_input('Adult Mortality', min_value=0, max_value=1000, value=263)
    income_composition_of_resources = st.sidebar.number_input('Income Composition of Resources', min_value=0.0, max_value=1.0, value=0.479)
    schooling = st.sidebar.number_input('Schooling', min_value=0.0, max_value=20.7, value=10.1)
    hiv_aids = st.sidebar.number_input('HIV/AIDS', min_value=0.1, max_value=50.6, value=0.7)
    total_expenditure = st.sidebar.number_input('Total Expenditure', min_value=0.37, max_value=17.6, value=8.16)
    # population = st.sidebar.number_input('Population', min_value=0, max_value=int(df['Population'].max()), value=33736494)
    bmi = st.sidebar.number_input('BMI', min_value=1.0, max_value=87.3, value=33.1)
    # measles = st.sidebar.number_input('Measles', min_value=0, max_value=int(df['Measles'].max()), value=1154)
    under_fivedeaths = st.sidebar.number_input('Under-five Deaths', min_value=0.0, max_value=2500.0, value=50.0)
    alcohol = st.sidebar.number_input('Thinness 5 to 9 years old', min_value=0.01, max_value=17.87, value=15.0)
    thinness5_9years = st.sidebar.number_input('alcohol', min_value=0.1, max_value=28.6, value=19.1)
    
    data = {
        'AdultMortality': adult_mortality,
        'Incomecompositionofresources': income_composition_of_resources,
        'Schooling': schooling,
        'Totalexpenditure': total_expenditure,
        'HIV/AIDS': hiv_aids,
        'BMI': bmi,
        'under-fivedeaths': under_fivedeaths,
        'thinness5-9years' : thinness5_9years,
        'Alcohol': alcohol
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Create a copy of the input data for display purposes
display_input_df = input_df.copy()

# # Encode the input data
# categorical_features = []
# # input_data[categorical_features] = encoder.transform(input_data[categorical_features])
# input_df[categorical_features] = encoder.transform(input_df[categorical_features])

st.subheader('User Input Parameters')
st.write(display_input_df)

important_rf_features = ['under-fivedeaths', 'AdultMortality', 'Incomecompositionofresources', 'Schooling', 'HIV/AIDS', 'Totalexpenditure', 'thinness5-9years', 'BMI', 'Alcohol']

# Predict the life expectancy
prediction = rf_model.predict(input_df[important_rf_features])


st.subheader('Prediction')
st.write(f"Predicted Life Expectancy: {prediction[0]:.2f}")
