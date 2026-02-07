import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Demand Forecasting", layout="centered")

st.title("📦 AI-Based Demand Forecasting System")

st.write(
    "This web application forecasts short-term product demand "
    "using historical sales data to support smart inventory decisions."
)

uploaded_file = st.file_uploader("Upload Sales Data (CSV file)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Sales Data")
    st.dataframe(df.head())

    df = df.rename(columns={"Date": "ds", "Quantity": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    st.subheader("📈 Demand Forecast Visualization")
    fig = model.plot(forecast)
    st.pyplot(fig)

    st.subheader("🔮 Predicted Demand (Next 7 Days)")
    st.dataframe(forecast[["ds", "yhat"]].tail(7))
