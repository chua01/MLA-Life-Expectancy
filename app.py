import streamlit as st
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('led.csv')

# Handling missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical features using Target Encoder
categorical_features = ['Country']
encoder = TargetEncoder(cols=categorical_features)
df[categorical_features] = encoder.fit_transform(df[categorical_features], df['Lifeexpectancy'])

# Define the important features
important_rf_features = ['Country', 'Year', 'AdultMortality', 'Incomecompositionofresources', 'Schooling', 'HIV/AIDS', 'Totalexpenditure', 'Population', 'BMI', 'Measles']

# Define the feature columns and target column
X_rf_important = df[important_rf_features]
y = df['Lifeexpectancy']

# Train Random Forest with top important features
rf_model_important = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_important.fit(X_rf_important, y)

# Function to predict life expectancy based on input data
def predict_life_expectancy(model, input_data, important_features):
    input_data = input_data[important_features]
    return model.predict(input_data)

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
input_df[categorical_features] = encoder.transform(input_df[categorical_features])

# Predict the life expectancy
prediction = predict_life_expectancy(rf_model_important, input_df, important_rf_features)

st.subheader('User Input Parameters')
st.write(display_input_df)

st.subheader('Prediction')
st.write(f"Predicted Life Expectancy: {prediction[0]:.2f}")
