import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = {"store", "date"}

@st.cache_data
def load_data(uploaded_file):
    
    #  Ensure correct file type
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Please upload a valid CSV file.")

    # Load the CSV
    df = pd.read_csv(uploaded_file)

    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df
