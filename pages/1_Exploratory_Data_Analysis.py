import streamlit as st
import pandas as pd
from src.visualization import plot_sales_trend, plot_prediction_vs_actual


st.sidebar.image("media/Yoobic.png", width='stretch')
st.title("Exploratory Data Analysis")

uploaded_file = st.file_uploader("Upload CSV for EDA", type=["csv"], key="eda_upload")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        if "store" not in df.columns or "weekly_sales" not in df.columns:
            st.error("The dataset must contain 'store' and 'weekly_sales' columns.")
        else:
            stores = sorted(df["store"].unique())
            selected_store = st.selectbox("Select Store", stores)

            store_df = df[df["store"] == selected_store]

            st.subheader(f"Weekly Sales Trend for Store {selected_store}")
            st.pyplot(plot_sales_trend(store_df))

            if "predicted_weekly_sales" in df.columns:
                st.subheader("Actual vs Predicted Comparison")
                st.pyplot(plot_prediction_vs_actual(df))

    except Exception as e:
        st.error(f"Unable to load or analyze data: {e}")
