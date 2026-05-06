import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data['Gender'] = label_encoder.fit_transform(data['Gender'])

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

encoded_geo = encoder.fit_transform(data[['Geography']])

df_encoded = pd.DataFrame(
    encoded_geo.toarray(),
    columns=encoder.get_feature_names_out()
)

data = data.reset_index(drop=True)
df_encoded = df_encoded.reset_index(drop=True)

data = pd.concat([data, df_encoded], axis=1)

data = data.drop('Geography', axis=1)

# save encoders
import pickle

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

# divide independent and dependent features
x = data.drop('Exited', axis=1)
y = data['Exited']

# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# ANN IMPLEMENTATION
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

import datetime

from tensorflow.keras.layers import Dense, Input

model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(
    optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# early stopping
earlystopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# train model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=100,
    callbacks=[tensorboard_callback, earlystopping_callback]
)

# save model
model.save('model.keras')