import streamlit as st
import pandas as pd

from src.dataloader import load_data
from src.preprocessing import preprocess_input
from src.inference import load_model, predict
from src.visualization import plot_actual_vs_predicted_time
from utils.helper_metrics import evaluate_model


st.set_page_config(page_title="Store Sales Prediction", layout="wide")
st.sidebar.image("media/Yoobic.png", width='stretch')

st.title("Store Sales Prediction App")
st.write("Upload a CSV dataset to generate weekly sales predictions.")


# -------------------------
# SESSION STATE INIT
# -------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "model" not in st.session_state:
    st.session_state.model = None


# -------------------------
# MODEL SELECTION
# -------------------------
model_choice = st.selectbox(
    "Select Model",
    (
        "Linear Regression",
        "Linear Regression All data",
        "Random Forest",
        "Random Forest All data",
        "XGBoost",
        "XGBoost All data",
    )
)

model_paths = {
    "Linear Regression": "models/linear_regression.pkl",
    "Linear Regression All data": "models/linear_regression_all.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Random Forest All data": "models/random_forest_all.pkl",
    "XGBoost": "models/xgboost.pkl",
    "XGBoost All data": "models/xgboost_all.pkl",
}


if st.button("Load Model"):
    try:
        st.session_state.model = load_model(model_paths[model_choice])
        st.success(f"{model_choice} model loaded successfully.")
    except Exception as e:
        st.error(f"Model loading failed: {e}")


# -------------------------
# DATA UPLOAD & CACHING
# -------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        df = load_data(uploaded_file)

        required = {"store", "date"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            st.error(f"Uploaded file missing required columns: {missing}")
        else:
            st.session_state.df = df
            st.success("Dataset loaded and cached successfully.")

    except Exception as e:
        st.error(f"Error reading file: {e}")


# -------------------------
# DISPLAY DATA IF EXISTS
# -------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Dataset Preview")
    st.dataframe(df.head())


    # ---------------------
    # PREDICT BUTTON
    # ---------------------
    if st.button("Predict Weekly Sales"):
        if st.session_state.model is None:
            st.error("Please load a model first.")
        else:
            store_stats = pd.read_csv("data/store_stats.csv")
            df_pre = preprocess_input(df, store_stats)
            df_pre = predict(df_pre, st.session_state.model)

            st.session_state.predictions = df_pre
            st.success("Predictions generated and stored.")


# -------------------------
# SHOW PREDICTIONS
# -------------------------
if st.session_state.predictions is not None:
    df_pred = st.session_state.predictions

    st.subheader("Predicted Sales")
    st.dataframe(df_pred[["store", "date", "predicted_weekly_sales"]])

    st.download_button(
        "Download Predictions",
        df_pred.to_csv(index=False).encode("utf-8"),
        "predicted_sales.csv"
    )

    # -------------------------
    # METRICS (only if actual exists)
    # -------------------------
    if "weekly_sales" in df_pred.columns:
        mae, rmse, mape = evaluate_model(df_pred["weekly_sales"], df_pred["predicted_weekly_sales"])

        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:,.2f}")
        col2.metric("RMSE", f"{rmse:,.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")

        stores = sorted(df_pred["store"].unique())
        selected_store = st.selectbox("Select Store For Plot", stores)

        store_df = df_pred[df_pred.store == selected_store]

        st.subheader("Actual vs Predicted Trend")
        st.pyplot(
            plot_actual_vs_predicted_time(
                store_df["weekly_sales"],
                store_df["predicted_weekly_sales"],
                model=model_choice
            )
        )
