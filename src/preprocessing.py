import pandas as pd


superbowl = [pd.Timestamp("2010-02-12"),pd.Timestamp("2011-02-11"),pd.Timestamp("2012-02-10") ]
labour_day = [ pd.Timestamp("2010-09-10"),pd.Timestamp("2011-09-9"),pd.Timestamp("2012-09-07")]
thanksgiving = [pd.Timestamp("2010-11-26"),pd.Timestamp("2011-11-25"),pd.Timestamp("2012-11-23")]
christmas = [pd.Timestamp("2010-12-31"),pd.Timestamp("2011-12-30"),pd.Timestamp("2012-12-28")] 


def preprocess_input(df):
    df = df.copy()
    # Convert date to datetime and engineer date features
    df['date'] = pd.to_datetime(df['date'], format = '%d-%m-%Y')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    df['is_superbowl']    = df['date'].isin(superbowl).astype(int)
    df['is_laborday']     = df['date'].isin(labour_day).astype(int)
    df['is_thanksgiving'] = df['date'].isin(thanksgiving).astype(int)
    df['is_christmas']    = df['date'].isin(christmas).astype(int)
    
    df = df.sort_values('date')

    # Create lagged features
    df ['weekly_sales_lag_1'] = df.groupby('store')['weekly_sales'].shift(1)
    df['weekly_sales_lag_4'] = df.groupby('store')['weekly_sales'].shift(4)
    
    df['rolling_mean_4'] = df.groupby('store')['weekly_sales'].transform(lambda x: x.rolling(4).mean())
    
    # Logic to fill NA values for prediction 
    store_mean = df.groupby('store')['weekly_sales'].transform('mean')
    global_mean = df['weekly_sales'].mean()

    df['weekly_sales_lag_1'] = df['weekly_sales_lag_1'].fillna(store_mean).fillna(global_mean)
    df['weekly_sales_lag_4'] = df['weekly_sales_lag_4'].fillna(store_mean).fillna(global_mean)
    df['rolling_mean_4'] = df['rolling_mean_4'].fillna(store_mean).fillna(global_mean)
    
    df = df.reset_index(drop=True)
    return df