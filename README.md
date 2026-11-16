# Bike Sharing Demand Prediction

**Machine Learning Project | Regression | Time Series Analysis | Predictive Analytics**

## 1. Project Overview

The Bike Sharing Demand Prediction project aims to develop machine learning models capable of forecasting bicycle rental demand based on historical rental records, temporal patterns, seasonal variations, and weather conditions.

The project uses the publicly available Bike Sharing Dataset from the UCI Machine Learning Repository, which contains daily and hourly bicycle rental records collected during 2011 and 2012.

The primary objective is to identify the factors influencing bicycle rental demand and develop predictive models that can support operational planning, resource allocation, and demand forecasting.

## 2. Business Problem

Bike-sharing systems must continuously balance bicycle availability with customer demand.

Insufficient bicycle availability can result in lost rental opportunities, while excessive inventory can increase operational and maintenance costs.

Accurate demand forecasting can help organizations:

- Optimize bicycle distribution across stations.
- Anticipate peak rental periods.
- Improve resource allocation.
- Support maintenance scheduling.
- Identify seasonal and weather-related demand patterns.
- Improve operational efficiency and customer satisfaction.

This project explores how historical data and machine learning techniques can help address these challenges.

## 3. Project Objectives

### Primary Objective

Develop regression-based machine learning models to predict the number of bicycle rentals using historical and contextual information.

### Specific Objectives

1. Explore historical bicycle rental data.
2. Identify temporal, seasonal, and meteorological demand patterns.
3. Perform data cleaning and preprocessing.
4. Engineer relevant predictive features.
5. Train and evaluate multiple regression algorithms.
6. Compare model performance using statistical evaluation metrics.
7. Identify the most relevant factors influencing bicycle rental demand.
8. Establish a foundation for future deployment as a predictive service.

## 4. Dataset Description

The project uses the Bike Sharing Dataset published by the UCI Machine Learning Repository.

**Dataset source:**  
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

The dataset includes two primary files:

| Dataset | Records | Description |
|---|---:|---|
| `day.csv` | 731 | Daily bicycle rental records |
| `hour.csv` | 17,379 | Hourly bicycle rental records |

### 4.1 Daily Dataset

The daily dataset contains aggregated rental information and environmental variables.

### 4.2 Hourly Dataset

The hourly dataset provides more granular information, allowing the analysis of demand fluctuations throughout the day.

### 4.3 Main Variables

| Variable | Description |
|---|---|
| `dteday` | Date of the observation |
| `season` | Season of the year |
| `yr` | Year indicator |
| `mnth` | Month |
| `hr` | Hour of the day (hourly dataset) |
| `holiday` | Holiday indicator |
| `weekday` | Day of the week |
| `workingday` | Working-day indicator |
| `weathersit` | Weather conditions |
| `temp` | Normalized temperature |
| `atemp` | Normalized feeling temperature |
| `hum` | Normalized humidity |
| `windspeed` | Normalized wind speed |
| `casual` | Number of casual users |
| `registered` | Number of registered users |
| `cnt` | Total bicycle rentals |

**Target variable:** `cnt`

Important: `cnt` is calculated as the sum of `casual` and `registered`. These two variables must therefore be excluded from predictive features when forecasting total rentals to avoid target leakage.

## 5. Methodology

The project follows a structured machine learning workflow.

### 5.1 Data Exploration

Exploratory Data Analysis (EDA) is performed to understand:

- Dataset structure and data types.
- Missing values and duplicate records.
- Distribution of bicycle rentals.
- Correlations between variables.
- Seasonal demand variations.
- Hourly rental patterns.
- Relationships between weather conditions and demand.

### 5.2 Data Preprocessing

The preprocessing stage includes:

- Data validation.
- Missing-value analysis.
- Duplicate detection.
- Feature selection.
- Categorical variable encoding when necessary.
- Numerical feature scaling where appropriate.

Daily and hourly datasets must be handled independently or joined using the date field (`dteday`) with a validated many-to-one relationship. Their `instant` columns are separate record identifiers and must not be treated as common business keys.

### 5.3 Feature Engineering

Feature engineering aims to improve predictive performance by extracting useful information from existing variables.

Potential engineered features include:

| Feature | Description |
|---|---|
| `is_peak_hour` | Identifies predefined peak commuting hours |
| `is_weekend` | Identifies weekend observations |
| `is_rush_hour` | Identifies morning and evening commuting periods |
| `month` | Represents monthly seasonality |
| `day_of_year` | Captures annual temporal patterns |
| `temperature_category` | Groups observations by temperature ranges |

Additional features should be validated experimentally to determine whether they improve generalization.

### 5.4 Model Development

The project explores regression algorithms for predicting bicycle rental demand.

Models of interest include:

- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Ridge Regression provides a regularized linear baseline, while tree-based algorithms can capture nonlinear relationships between environmental conditions, temporal variables, and demand.

### 5.5 Model Evaluation

Model performance should be evaluated using:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted rental counts.

**Root Mean Squared Error (RMSE)**

Penalizes larger prediction errors more heavily.

**Coefficient of Determination (R²)**

Measures the proportion of target variance explained by the model.

**Mean Absolute Percentage Error (MAPE)**

Measures relative prediction error, with special handling for actual values equal to zero.

For forecasting future demand, chronological training, validation, and test splits should be used rather than random splitting.

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| Jupyter Notebook | Interactive experimentation |
| XGBoost | Advanced gradient boosting (proposed) |
| MLflow | Experiment tracking (proposed) |
| FastAPI | Model deployment (proposed) |
| Power BI | Business intelligence (proposed) |

## 7. Project Structure

A recommended structure for the next version of the repository is:

```text
PREDICCION_BICICLETAS_ALQUILADAS/
│
├── data/
│   ├── raw/
│   │   ├── day.csv
│   │   └── hour.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── bike_sharing_analysis.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── api/
│   └── main.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── LICENSE
```

This is a proposed organization for future development, not a claim that every directory currently exists in the repository.

## 8. Installation and Execution

### Prerequisites

- Python 3.10 or later.
- pip.
- Jupyter Notebook or JupyterLab.

### Clone the Repository

```bash
git clone https://github.com/michaelventura01/PREDICCION_BICICLETAS_ALQUILADAS.git

cd PREDICCION_BICICLETAS_ALQUILADAS
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and execute the analysis sequentially.

## 9. Expected Results

The project aims to produce:

- An exploratory analysis of bicycle rental demand.
- Identification of relevant demand drivers.
- Trained regression models.
- Comparative model evaluation.
- Demand predictions based on temporal and weather conditions.
- Insights supporting operational decision-making.

Actual model performance should be documented after running the corrected preprocessing and evaluation pipeline.

## 10. Limitations

The original dataset has several limitations:

1. It covers only two calendar years (2011–2012).
2. It represents a specific bike-sharing system.
3. It may not generalize directly to other cities or transportation systems.
4. Weather and seasonal patterns may differ across geographical locations.
5. Historical patterns may change over time.
6. External factors such as public transportation disruptions and special events are not comprehensively represented.

These limitations should be considered when interpreting predictions and evaluating real-world applicability.

## 11. Future Improvements

Potential development directions include:

- Time-series forecasting models.
- Hyperparameter optimization.
- Advanced feature engineering.
- Model explainability using SHAP.
- Experiment tracking with MLflow.
- Automated model retraining.
- REST API deployment with FastAPI.
- Demand forecasting dashboards with Power BI.
- Integration with real-time weather data.
- Model performance and data drift monitoring.

## 12. Conclusion

The Bike Sharing Demand Prediction project demonstrates how historical operational data and machine learning techniques can be applied to transportation demand forecasting.

By combining temporal, seasonal, and meteorological information, predictive models can help identify demand patterns and support more efficient resource allocation.

The project provides a foundation for developing a more comprehensive predictive analytics solution, evolving from exploratory data analysis into a reproducible and deployable machine learning application.

## 13. References

1. Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence, 2*, 113–127.
2. UCI Machine Learning Repository. Bike Sharing Dataset. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

## Author

**Michael Ventura**  
Solutions Architect | Data Engineering | Machine Learning | Predictive Analytics

GitHub: https://github.com/michaelventura01/PREDICCION_BICICLETAS_ALQUILADAS
