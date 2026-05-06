import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
import streamlit as st

# load the trained model
model = tf.keras.models.load_model('model.keras', compile=False)

# load label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# load one hot encoder
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title("Customer Churn Prediction")

# input fields
geography = st.selectbox('Geography', encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare input data
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': label_encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# convert input to dataframe
input_df = pd.DataFrame([input_data])

# one hot encode geography
encoded_geo = encoder.transform(input_df[['Geography']])

encoded_geo_df = pd.DataFrame(
    encoded_geo.toarray(),
    columns=encoder.get_feature_names_out()
)

# remove geography column
input_df = input_df.drop('Geography', axis=1)

# combine encoded geography columns
input_df = pd.concat(
    [
        input_df.reset_index(drop=True),
        encoded_geo_df.reset_index(drop=True)
    ],
    axis=1
)

# scale input data
input_scaled = scaler.transform(input_df)

# prediction
if st.button('Predict'):

    prediction = model.predict(input_scaled)

    prediction_proba = prediction[0][0]

    st.write(f'Prediction Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.write("Customer is likely to churn")
    else:
        st.write("Customer is not likely to churn")