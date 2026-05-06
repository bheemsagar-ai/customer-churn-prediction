# Customer Churn Prediction

A deep learning web app that predicts whether a bank customer is likely to churn, built with TensorFlow and Streamlit.

## Overview

This project trains an Artificial Neural Network (ANN) on the [Churn Modelling dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) and serves predictions through an interactive Streamlit UI.

## Project Structure

```
├── data.py               # Data preprocessing, model training & saving
├── app.py                # Streamlit web app
├── model.keras           # Trained ANN model
├── encoder.pkl           # OneHotEncoder for Geography
├── label_encoder.pkl     # LabelEncoder for Gender
├── scaler.pkl            # StandardScaler for feature scaling
├── Churn_Modelling.csv   # Dataset
└── requirements.txt      # Dependencies
```

## Model Architecture

- Input layer → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
- Optimizer: Adam (lr=0.01)
- Loss: Binary Crossentropy
- Callbacks: EarlyStopping, TensorBoard

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train the model
```bash
python data.py
```

### Run the app
```bash
streamlit run app.py
```

## Input Features

| Feature | Description |
|---|---|
| Credit Score | Customer's credit score |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | 18 – 92 |
| Tenure | Years with the bank (0–10) |
| Balance | Account balance |
| Num of Products | Number of bank products (1–4) |
| Has Credit Card | 0 or 1 |
| Is Active Member | 0 or 1 |
| Estimated Salary | Annual salary estimate |

## Output

The app returns a churn probability score and classifies the customer as:
- **Likely to churn** — probability > 0.5
- **Not likely to churn** — probability ≤ 0.5
