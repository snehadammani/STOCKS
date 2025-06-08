import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    with open("stocks_dl.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Stock Prediction", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

uploaded_file = st.file_uploader("Upload your CSV stock data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    input_features = st.multiselect("Select input features", df.columns.tolist(), default=df.columns[:-1])

    if input_features:
        try:
            X = df[input_features]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            if st.button("🚀 Predict"):
                predictions = model.predict(X_scaled)
                predictions = predictions.flatten() if predictions.ndim > 1 else predictions
                st.write("### Predictions", predictions)
                st.line_chart(predictions)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
