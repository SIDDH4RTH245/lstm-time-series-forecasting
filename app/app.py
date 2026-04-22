import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "lstm_model.h5")
data_path = os.path.join(BASE_DIR, "data", "airline-passengers.csv")

model = tf.keras.models.load_model(model_path, compile=False)

df = pd.read_csv(data_path)
df.columns = ["Month", "Passengers"]
historical_data = df["Passengers"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(historical_data)

st.title("📈 LSTM Time Series Forecasting")

st.write("Enter last 10 months passenger data")

inputs = []

for i in range(10):
    val = st.number_input(f"Month {i+1}", value=100.0)
    inputs.append(val)

if st.button("Predict Next Month"):
    data = np.array(inputs).reshape(-1,1)
    data_scaled = scaler.transform(data)
    
    data_scaled = data_scaled.reshape(1,10,1)
    
    prediction = model.predict(data_scaled)
    
    prediction = scaler.inverse_transform(prediction)
    
    st.success(f"Next Month Prediction: {prediction[0][0]:.2f}")