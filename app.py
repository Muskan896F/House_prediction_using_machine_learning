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
st.write("### Input Features to Predict House Price")

# Initialize default values for features
CRIM = st.number_input('Per capita crime rate', value=float(df['crim'].mean()))
ZN = st.number_input('Proportion of residential land', value=float(df['zn'].mean()))
INDUS = st.number_input('Non-retail business acres', value=float(df['indus'].mean()))
CHAS = st.selectbox('Charles River dummy variable', (0, 1))
NOX = st.number_input('Nitric oxide concentration', value=float(df['nox'].mean()))
RM = st.number_input('Average number of rooms', value=float(df['rm'].mean()))
AGE = st.number_input('Proportion of older units', value=float(df['age'].mean()))
DIS = st.number_input('Distance to employment centers', value=float(df['dis'].mean()))
RAD = st.number_input('Accessibility to highways', value=float(df['rad'].mean()))
TAX = st.number_input('Property-tax rate', value=float(df['tax'].mean()))
PTRATIO = st.number_input('Pupil-teacher ratio', value=float(df['ptratio'].mean()))
B = st.number_input('Proportion of blacks', value=float(df['b'].mean()))
LSTAT = st.number_input('Lower status population', value=float(df['lstat'].mean()))

# Prepare input data for prediction
user_input = np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]).reshape(1, -1)

# Prediction Button to trigger the price prediction
if st.button('Predict Price'):
    predicted_price = house_predictor.predict(user_input)

    # Display the prediction
    st.write(f"### Predicted Home Price: ${predicted_price[0] * 1000:.2f}")