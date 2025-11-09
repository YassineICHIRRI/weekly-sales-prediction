import joblib

def load_model(model_path, scaler_path=None):
    model = joblib.load(model_path)
    return model

def predict(df, model):
    features = ['rolling_mean_4', 'weekly_sales_lag_1', 'weekly_sales_lag_4','month','store','is_superbowl', 'is_laborday','is_thanksgiving','is_christmas']
    X = df[features]

    df['predicted_weekly_sales'] = model.predict(X)
    return df
