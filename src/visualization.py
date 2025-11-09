import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_trend(df):
    
    '''
    Plot sales trends on average 
    '''
    fig, ax = plt.subplots(figsize=(18,9))

    
    # Plot average weekly sales over time
    df.groupby('date')['weekly_sales'].mean().plot(ax=ax, label='Average Weekly Sales')
    
    # Titles and labels
    ax.set_title("Average Weekly Sales Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")
    
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_actual_vs_predicted_time( y_test, y_pred, model="model"):
    """
    Plots actual vs predicted values over time.
    """
    fig, ax = plt.subplots(figsize=(18,9))
    ax.plot(y_test.index, y_test.values, label='Actual Sales')
    ax.plot(y_test.index, y_pred, label='Predicted Sales', linestyle='--')
    ax.set_title(f'Actual vs Predicted Weekly Sales for {model}')
    ax.set_ylabel("Weekly Sales")
    ax.legend()
    ax.grid(True)
    return fig
    

def plot_prediction_vs_actual(df):
    '''
    Scatter prediction relaive of actual sales values
    '''
    fig, ax = plt.subplots()
    ax.figure(figsize=(12,6))
    ax.scatter(df['weekly_sales'], df['predicted_weekly_sales'])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual Sales")
    
    return fig


def plot_store_variation(df):
    '''
    Plot a boxplot of store sales variations
    '''
    
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.boxplot(data = df, y= "weekly_sales", x = 'store', palette='viridis')
    ax.set_title("Weekly Sales Distribution by Store")
    ax.set_ylabel("Weekly Sales")
    ax.set_xlabel("Store ID")
    plt.xticks(rotation=90)
    plt.tight_layout()
    return fig
