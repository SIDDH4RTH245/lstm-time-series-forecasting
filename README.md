# 📈 LSTM Time Series Forecasting Project

## 📌 Overview
This project uses a **Long Short-Term Memory (LSTM)** neural network to forecast future values in a time series dataset.

The model learns patterns from historical data and predicts future trends using deep learning techniques.

---

## 🎯 Objective
- Understand time series data
- Create sequences for LSTM input
- Build and train an LSTM model
- Forecast future values
- Deploy model using Streamlit

---

## 📊 Dataset
- Dataset: Airline Passengers Dataset
- Monthly passenger data over time

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Converted date column
- Set time index
- Normalized data using MinMaxScaler

### 2️⃣ Sequence Creation
- Converted time series into supervised learning format
- Created input-output sequences for LSTM

### 3️⃣ Model Building
- Built LSTM model using TensorFlow/Keras
- Added Dense layers for prediction

### 4️⃣ Model Training
- Trained model on sequential data
- Optimized using Adam optimizer

### 5️⃣ Model Evaluation
- Compared predicted vs actual values
- Visualized results

### 6️⃣ Model Saving
- Saved trained model as `.h5` file

### 7️⃣ Deployment
- Built Streamlit app for forecasting

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- TensorFlow / Keras
- Scikit-learn
- Streamlit

---
