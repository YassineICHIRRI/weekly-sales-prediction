import streamlit as st
import pandas as pd
from src.preprocessing import preprocess_input
from src.inference import load_model, predict
from src.visualization import plot_actual_vs_predicted_time

st.sidebar.image("media/Yoobic.png",width='stretch')

st.set_page_config(page_title="Store Sales Prediction", layout="wide")

st.title("Store Sales Prediction")

st.write("Upload a CSV dataset containing store-level data to generate weekly sales predictions.")

if "model" not in st.session_state:
    st.session_state.model = None

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ("Linear Regression","Linear Regression All data", "Random Forest","Random Forest All data", "XGBoost", "XGBoost All data")
)

model_paths = {
    "Linear Regression": "models/linear_regression.pkl",
    "Linear Regression All data": "models/linear_regression_all.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Random Forest All data": "models/random_forest_all.pkl",
    "XGBoost": "models/xgboost.pkl",
    "XGBoost All data": "models/xgboost_all.pkl"
}

if st.button("Load Model"):
    try:
        st.session_state.model = load_model(model_paths[model_choice])
        st.success(f"{model_choice} model loaded successfully.")
    except Exception as e:
        st.error(f"Model loading failed: {e}")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        required_columns = {"store", "date"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            st.error(f"Missing required columns: {missing}")
        else:
            # After prediction
            if st.button("Predict Weekly Sales"):
                if st.session_state.model is None:
                    st.error("Please load a model first.")
                else:
                    df = preprocess_input(df)
                    df = predict(df, st.session_state.model)
                    
                    # Store predictions in session_state so they persist
                    st.session_state.predictions = df

            # Now, outside the button block, show predictions and plot
            if "predictions" in st.session_state:
                df_pred = st.session_state.predictions
                st.subheader("Predictions")
                st.dataframe(df_pred[['store', 'date', 'predicted_weekly_sales']])

                csv_data = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions", csv_data, "predicted_sales.csv")
                
                if 'weekly_sales' in df_pred.columns:
                    stores = sorted(df_pred["store"].unique())
                    selected_store = st.selectbox("Select Store", stores)
                    store_df = df_pred[df_pred["store"] == selected_store]
                    
                    st.subheader("Actual vs Predicted Comparison")
                    st.pyplot(plot_actual_vs_predicted_time(
                        store_df['weekly_sales'], 
                        store_df['predicted_weekly_sales'], 
                        model=model_choice
                    ))
    except Exception as e:
        st.error(f"Error processing file: {e}")
        

