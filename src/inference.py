import joblib


def load_model(model_path):
    '''
    Load the ML model from a .pkl file
    
    Args:
        model_path : Path to the model 
        
    Return :
        model : Loaded Model
    '''
    model = joblib.load(model_path)
    return model


def predict(df, model):
    '''
    Running predictions from model
    
    Args:
        df : Test Dataset
        model: loaded model 
        
    Return :
        df : Dataset with predictions 
    '''
    
    features = ['rolling_mean_4', 'weekly_sales_lag_1', 'weekly_sales_lag_4','month','store','is_superbowl', 'is_laborday','is_thanksgiving','is_christmas']
    X = df[features]

    df['predicted_weekly_sales'] = model.predict(X)
    return df
