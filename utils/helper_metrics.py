import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def RMSE(y_test, y_pred):
    metric = np.sqrt(mean_squared_error(y_test, y_pred))
    return metric


def mape(y_test,y_pred):
    value = np.mean(np.abs((y_test - y_pred ) / y_test)) * 100
    return value


def evaluate_model(y_true, y_pred):
    """
    Computes MAE, RMSE, and MAPE for predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape