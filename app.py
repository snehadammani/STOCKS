import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    with open("stocks_dl.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("📈 Stock Price Prediction Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Uploaded Data Preview", df.head())

    # Assume last column is target (optional)
    X = df.iloc[:, :-1].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if st.button("Predict"):
        try:
            predictions = model.predict(X_scaled)
            st.write("### 📊 Predictions:")
            st.write(predictions)

            # Plot
            st.line_chart(predictions)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Upload a CSV file to start prediction.")
