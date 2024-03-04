import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('scoring_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess input data and make predictions
def predict_scoring(features):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([features])
    # Perform any necessary preprocessing (e.g., encoding categorical variables)
    # Ensure that the order of features matches the order used during model training
    # For simplicity, assuming one-hot encoding for categorical variables

    # Make predictions
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title('Scoring Prediction App')

    # Sidebar with user input
    st.sidebar.header('User Input Features')

    # Example input features (replace with your input fields)
    tier = st.sidebar.selectbox('Tier', ['T1', 'T2', 'T3'])
    application_type = st.sidebar.selectbox('Application Type', ['COTS', 'Homegrown'])
    hosting_type = st.sidebar.selectbox('Hosting Type', ['On-Premise', 'AWS', 'PrivateCloud'])
    # Add more input features as needed

    # Collect user input features
    user_input = {
        'Tier': tier,
        'Application Type': application_type,
        'Hosting Type': hosting_type,
        # Add more input features as needed
    }

    # Make prediction
    scoring_prediction = predict_scoring(user_input)

    # Display prediction
    st.write(f'Predicted Scoring: {scoring_prediction}')

if __name__ == '__main__':
    main()
