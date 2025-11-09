import matplotlib.pyplot as plt

def plot_sales_trend(df):
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

def plot_prediction_vs_actual(df):
    fig, ax = plt.subplots()
    ax.figure(figsize=(12,6))
    ax.scatter(df['weekly_sales'], df['predicted_weekly_sales'])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual Sales")
    
    return fig
