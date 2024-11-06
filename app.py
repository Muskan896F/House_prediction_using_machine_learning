import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Title of the Web Page
st.title("Boston Housing Price Prediction")

# Description of the Project
st.write("""
    This app predicts the median home values in Boston based on various features such as crime rate, number of rooms,
    property-tax rate, etc. It uses a linear regression model trained on the Boston Housing dataset.
""")

# Load and Show Data
df = pd.read_csv("BostonHousing.csv")
st.write("### Dataset Overview", df.head())

# Data Preprocessing
X = df.drop(["medv"], axis=1)  # Features (Independent variables)
y = df["medv"]  # Target variable (Dependent)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
house_predictor = LinearRegression()
house_predictor.fit(X_train, y_train)
y_pred = house_predictor.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.write(f"### Model Performance")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")

# Allow User to Input Data for Prediction
st.write("### Predict House Price")
CRIM = st.slider('Per capita crime rate', float(df['crim'].min()), float(df['crim'].max()), float(df['crim'].mean()))
ZN = st.slider('Proportion of residential land', float(df['zn'].min()), float(df['zn'].max()), float(df['zn'].mean()))
INDUS = st.slider('Non-retail business acres', float(df['indus'].min()), float(df['indus'].max()), float(df['indus'].mean()))
CHAS = st.radio('Charles River dummy variable', (0, 1))
NOX = st.slider('Nitric oxide concentration', float(df['nox'].min()), float(df['nox'].max()), float(df['nox'].mean()))
RM = st.slider('Average number of rooms', float(df['rm'].min()), float(df['rm'].max()), float(df['rm'].mean()))
AGE = st.slider('Proportion of older units', float(df['age'].min()), float(df['age'].max()), float(df['age'].mean()))
DIS = st.slider('Distance to employment centers', float(df['dis'].min()), float(df['dis'].max()), float(df['dis'].mean()))
RAD = st.slider('Accessibility to highways', float(df['rad'].min()), float(df['rad'].max()), float(df['rad'].mean()))
TAX = st.slider('Property-tax rate', float(df['tax'].min()), float(df['tax'].max()), float(df['tax'].mean()))
PTRATIO = st.slider('Pupil-teacher ratio', float(df['ptratio'].min()), float(df['ptratio'].max()), float(df['ptratio'].mean()))
B = st.slider('Proportion of blacks', float(df['b'].min()), float(df['b'].max()), float(df['b'].mean()))
LSTAT = st.slider('Lower status population', float(df['lstat'].min()), float(df['lstat'].max()), float(df['lstat'].mean()))

# Prepare the input data for prediction
user_input = np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]).reshape(1, -1)

# Predict using the trained model
predicted_price = house_predictor.predict(user_input)

# Display the prediction
st.write(f"### Predicted Home Price: ${predicted_price[0] * 1000:.2f}")