import numpy as np 
import matplotlib.pyplot as plt 



def plot_predicted_vs_actual(y_test,y_pred, model = 'model'):
    
    # y_test and y_pred are the actual and predicted weekly sales
    
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', lw=2)  # perfect prediction line
    plt.xlabel("Actual Weekly Sales")
    plt.ylabel("Predicted Weekly Sales")
    plt.title(f"Predicted vs Actual Weekly Sales for {model}")
    plt.grid(True)
    plt.show()
    
    

def plot_actual_vs_predicted_time(df, y_test, y_pred, model="model"):
    """
    Plots actual vs predicted values over time.
    """

    plt.figure(figsize=(12,5))
    plt.plot(y_test.index, y_test.values, label='Actual Sales')
    plt.plot(y_test.index, y_pred, label='Predicted Sales', linestyle='--')
    plt.title(f'Actual vs Predicted Weekly Sales for {model}')
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_store_predictions(df, X_test, y_test, store_number, model, target_col='weekly_sales'):
    """
    Plots the actual vs predicted weekly sales for a given store.

    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe containing 'store' and 'date'.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True target values for the test set.
    store_number : int
        The store number to plot.
    model : trained model
        The model used for prediction (e.g., RandomForest).
    target_col : str
        Column name of actual sales in df.
    """
    # Filter test set for this store
    store_X = X_test[X_test['store'] == store_number]
    store_y = y_test[X_test['store'] == store_number]

    # Predict
    y_pred_store = model.predict(store_X)

    # Get corresponding dates from df
    store_dates = df.loc[store_X.index, 'date']

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(store_dates, store_y, label='Actual', marker='o')
    plt.plot(store_dates, y_pred_store, label='Predicted', marker='x')
    plt.title(f"Store {store_number}: Actual vs Predicted Weekly Sales")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    


def plot_rf_feature_importances(rf_model, feature_names, top_n=None, figsize=(10,6)):
    """
    Prints and plots Random Forest feature importances.

    Parameters:
    -----------
    rf_model : trained RandomForestRegressor
        The trained RF model.
    feature_names : list
        List of feature names corresponding to model input.
    top_n : int or None
        Number of top features to show. If None, show all.
    figsize : tuple
        Figure size for the plot.
    """
    # Compute importances
    importances = rf_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    if top_n:
        indices = indices[:top_n]

    # Print feature ranking
    print("Feature ranking:")
    for f, idx in enumerate(indices, start=1):
        print(f"{f}. {feature_names[idx]} ({importances[idx]:.4f})")

    # Plot
    plt.figure(figsize=figsize)
    plt.title("Random Forest Feature Importances")
    plt.bar(range(len(indices)), importances[indices], yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    plt.show()


def plot_xgb_feature_importances(model, features):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45)
    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()
    plt.show()
