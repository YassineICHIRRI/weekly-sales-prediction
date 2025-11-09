# Store Sales Prediction APP

A Streamlit-based web application to predict weekly sales for stores using machine learning models.    
The app supports multiple models (Linear Regression, Random Forest, XGBoost) and allows users to upload CSV datasets for predictions, visualize trends, and explore model performance.

---
The project includes:
- Exploratory Data Analysis (EDA) and model training in a Jupyter Notebook (`notebooks/EDA_Model_training.ipynb`).
- A Streamlit web app (`app.py`) for predictions, with features like model selection, visualizations, performance metrics, and interactivity.
- Pre-trained models saved in the `models/` directory.
- Helper scripts for data loading, preprocessing, inference, and visualization.
  
## Features

- Upload store-level CSV datasets for predictions
- Supports multiple ML models: Linear Regression, Random Forest, XGBoost
- Generates lag features and rolling means for improved predictions
- Displays predicted vs actual sales trends for each store
- Allows downloading predictions as CSV
- Exploratory Data Analysis (EDA) page with visualizations:
  - Weekly sales trend
  - Store-level sales variation
 
## Approach Summary
### Data Handling and Preprocessing
- Loaded the dataset `store_sales.csv` which spans from 2010-02-05 to 2012-11-01, containing sales data for 45 stores.
- Performed feature engineering: 
  - Extracted temporal features like `year` and `month` from the `date` column.
  - Created binary indicators for holidays (Super Bowl, Labor Day, Thanksgiving, Christmas) based on provided dates.
  - Added lag features (`weekly_sales_lag_1`, `weekly_sales_lag_4`) and rolling mean (`rolling_mean_4`) to capture time-series dependencies.
- Handled missing values by filling with store-specific or global means.

### Exploratory Data Analysis
In `EDA_Model_training.ipynb`, I conducted EDA to understand the data:
- **Dataset Overview**: Inspected structure. The dataset has 6435 rows with columns like `store`, `date`, `weekly_sales`, `holiday_flag`, `temperature`, `fuel_price`, `cpi`, and `unemployment`.
- **Sales Trends**: Plotted average weekly sales over time, revealing seasonal patterns (e.g., peaks during holidays) and variations across stores.
- **Store Variations**: Boxplots showed significant differences in sales distribution by store, indicating store ID as a key feature.
- **Correlations**: Heatmap analysis revealed weak correlations between external variables (e.g., temperature, fuel_price) and sales. Holidays had a noticeable impact.
- **Holiday Impact**: Analyzed sales spikes during holidays like Thanksgiving and Christmas.
- **Time-Series Insights**: Autocorrelation checks confirmed lag features would help model temporal dependencies.
- Insights guided feature selection: Prioritized lags, rolling means, month, store ID, and holiday indicators over less correlated features like temperature and unemployment (which were dropped after testing for low importance).

### Model Design Choices
- **Problem Framing**: Predicting the weekly sales value 
- **Models Trained**:
  - **Linear Regression**: Baseline model for simplicity and interpretability.
  - **Random Forest Regressor**: Chosen for handling non-linear relationships and feature interactions.
  - **XGBoost Regressor**: Selected for gradient boosting efficiency and handling time-series data.
  - 
- **Training Variants**: Trained models on all features and a set of important features, the second set gave the best results.
- **Evaluation Metrics**: Used MAE, RMSE, and MAPE. XGBoost achieved the lowest RMSE (~50,000-60,000 on test set), indicating good accuracy.
- **Reasoning**: Started with simple models and escalated to ensembles for better performance. Incorporated time-series features to address autocorrelation. Dropped low-importance features (e.g., temperature) based on EDA and model importances to reduce noise.
- **Model Saving**: Used `joblib` to save models for inference in the app.

> A detailed analysis is present in the notebook `notebooks/EDA_Model_training.ipynb`

## Installation

1. Clone the repository:
   ```bash
   https://github.com/YassineICHIRRI/weekly-sales-prediction.git
   cd weekly-sales-prediction
   ```

### Method 1: Using Virtual Environment (venv)

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit App :
   ```bash
   streamlit run app.py
   ```

### Method 2: Using Docker : 

> Ensure that docker is installed ( and Docker desktop is running if using Windows )
  
2. Build the docker image :
   ```cmd 
   docker build -t store-sales-app .
   ```
3. Run the docker container (exposes Streamlit on port 8501): 
```cmd
docker run -p 8501:8501 store-sales-app
```

