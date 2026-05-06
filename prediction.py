import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# load label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# load one hot encoder
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# load trained model
model = load_model('model.h5')

# example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

# convert input to dataframe
input_df = pd.DataFrame([input_data])

# label encode gender
input_df['Gender'] = label_encoder.transform(input_df['Gender'])

# one hot encode geography
encoded_geo = encoder.transform(input_df[['Geography']])

encoded_geo_df = pd.DataFrame(
    encoded_geo.toarray(),
    columns=encoder.get_feature_names_out()
)
print(encoded_geo_df)

# remove geography column
input_df = input_df.drop('Geography', axis=1)

# combine encoded geography columns
input_df = pd.concat([input_df.reset_index(drop=True),
                      encoded_geo_df.reset_index(drop=True)], axis=1)
print(input_df)

# scale input data
input_scaled = scaler.transform(input_df)

# prediction
prediction = model.predict(input_scaled)

# output
print("Prediction:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Customer is likely to churn")
else:
    print("Customer is not likely to churn")