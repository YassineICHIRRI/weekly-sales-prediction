import streamlit as st
import pandas as pd
from src.visualization import plot_sales_trend, plot_prediction_vs_actual, plot_store_variation

st.sidebar.image("media/Yoobic.png", width='stretch')
st.title("Exploratory Data Analysis")

# Ensure dataset is loaded from previous page
if "df" not in st.session_state:
    st.error("No dataset found. Please go back to the upload page and upload data first.")
    st.stop()

df = st.session_state.df
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Check required columns
required_cols = {"store", "weekly_sales"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"The dataset must contain the following columns: {missing}")
    st.stop()

# Select store
stores = sorted(df["store"].unique())
selected_store = st.selectbox("Select Store", stores)

store_df = df[df["store"] == selected_store]

# Plot weekly sales trend
st.subheader(f"Weekly Sales Trend for Store {selected_store}")
st.pyplot(plot_sales_trend(store_df))

# Plot store boxplots
st.subheader(f"Average sales boxplot")
st.pyplot(plot_store_variation(df))

# Plot prediction vs actual 
if "predicted_weekly_sales" in df.columns:
    st.subheader("Actual vs Predicted Comparison")
    st.pyplot(plot_prediction_vs_actual(store_df))
