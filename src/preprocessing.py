import pandas as pd

# Creating holidays arrays
superbowl = [pd.Timestamp("2010-02-12"),pd.Timestamp("2011-02-11"),pd.Timestamp("2012-02-10") ]
labour_day = [ pd.Timestamp("2010-09-10"),pd.Timestamp("2011-09-9"),pd.Timestamp("2012-09-07")]
thanksgiving = [pd.Timestamp("2010-11-26"),pd.Timestamp("2011-11-25"),pd.Timestamp("2012-11-23")]
christmas = [pd.Timestamp("2010-12-31"),pd.Timestamp("2011-12-30"),pd.Timestamp("2012-12-28")] 


def preprocess_input(df, store_stats=None):
    """Apply Preprocessing Pipeline

    Args:
        df (pd.Dataframe): original dataset
        store_stats (pd.Dataframe): Dataframe containing store-wise stats. Defaults to None.

    Returns:
        df: Preprocessed dataset
    """
    
    df = df.copy()

    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df['is_superbowl']    = df['date'].isin(superbowl).astype(int)
    df['is_laborday']     = df['date'].isin(labour_day).astype(int)
    df['is_thanksgiving'] = df['date'].isin(thanksgiving).astype(int)
    df['is_christmas']    = df['date'].isin(christmas).astype(int)

    df = df.sort_values(['store', 'date'])

    # Case 1: dataset contains weekly_sales → training or evaluation
    if "weekly_sales" in df.columns:
        df['weekly_sales_lag_1'] = df.groupby('store')['weekly_sales'].shift(1)
        df['weekly_sales_lag_4'] = df.groupby('store')['weekly_sales'].shift(4)
        df['rolling_mean_4'] = df.groupby('store')['weekly_sales'].transform(lambda x: x.rolling(4).mean())

        store_mean = df.groupby('store')['weekly_sales'].transform('mean')
        global_mean = df['weekly_sales'].mean()

        df['weekly_sales_lag_1'] = df['weekly_sales_lag_1'].fillna(store_mean).fillna(global_mean)
        df['weekly_sales_lag_4'] = df['weekly_sales_lag_4'].fillna(store_mean).fillna(global_mean)
        df['rolling_mean_4']     = df['rolling_mean_4'].fillna(store_mean).fillna(global_mean)

    # Case 2: dataset does NOT contain weekly_sales → prediction mode
    else:
        if store_stats is None:
            raise ValueError("weekly_sales column missing and no store_stats provided.")

        df = df.merge(store_stats, on="store", how="left")

        df.rename(columns={
            "lag1": "weekly_sales_lag_1",
            "lag4": "weekly_sales_lag_4",
            "roll4": "rolling_mean_4"
        }, inplace=True)

    return df.reset_index(drop=True)
