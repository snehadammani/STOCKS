import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load Keras model from pickle
@st.cache_resource
def load_model():
    with open("stocks_dl.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.set_page_config(page_title="Stock Prediction", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

st.markdown("""
This app allows you to upload a stock dataset, normalize it, and get predictions using a pre-trained deep learning model.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV stock data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Preview of Uploaded Data")
    st.write(df.head())

    # Feature selection
    st.subheader("🧮 Feature Configuration")
    all_columns = df.columns.tolist()
    input_features = st.multiselect("Select input features for prediction", all_columns, default=all_columns[:-1])

    if input_features:
        try:
            X = df[input_features]
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            if st.button("🚀 Predict"):
                predictions = model.predict(X_scaled)

                # Convert predictions to 1D array if needed
                if predictions.ndim > 1:
                    predictions = predictions.flatten()

                st.success("✅ Prediction Complete!")
                st.write("### 🔢 Predicted Output")
                st.dataframe(predictions)

                # Plotting predictions
                st.subheader("📊 Prediction Plot")
                fig, ax = plt.subplots()
                ax.plot(predictions, label="Predicted", color="green")
                ax.set_title("Predicted Stock Values")
                ax.set_xlabel("Time")
                ax.set_ylabel("Predicted Value")
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("Please select at least one input feature.")
else:
    st.info("⬆️ Upload a CSV file to start.")
