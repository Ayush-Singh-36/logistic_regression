# Logistic Regression for Australian Weather Prediction

## Overview
This notebook provides an end-to-end example of building and evaluating a binary classification model using Logistic Regression. The goal is to predict the `RainTomorrow` target variable, indicating whether it will rain on the next day, based on various meteorological features.

## Dataset
The model is trained on the `weatherAUS.csv` dataset, which contains daily weather observations from various locations across Australia. Key features include:
-   **Date**: Date of observation
-   **Location**: City/Region of observation
-   **MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine**: Various numerical weather measurements
-   **WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm**: Wind-related measurements
-   **Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm**: Humidity, pressure, and cloud cover observations
-   **Temp9am, Temp3pm**: Temperature at 9 AM and 3 PM
-   **RainToday**: Whether it rained today (categorical: 'Yes'/'No')
-   **RainTomorrow**: Whether it rained tomorrow (target variable: 'Yes'/'No')

## Features

### 1. Data Loading and Initial Exploration
-   Loads the `weatherAUS.csv` into a pandas DataFrame.
-   Provides an initial overview of the dataset using `df.info()` to check data types and non-null counts.

### 2. Data Cleaning and Preprocessing
-   **Missing Values Handling**: Drops rows where `RainToday` or `RainTomorrow` are missing to ensure target integrity. Numerical features with missing values are imputed using the mean strategy from `sklearn.impute.SimpleImputer`.
-   **Feature Scaling**: Numerical features are scaled using `sklearn.preprocessing.MinMaxScaler` to bring them into a common range (0-1), which helps in optimizing the model.
-   **Categorical Encoding**: Categorical features are converted into numerical format using One-Hot Encoding via `sklearn.preprocessing.OneHotEncoder`.

### 3. Data Splitting
-   The dataset is split into training, validation, and test sets based on the `Date` column:
    -   **Training Set**: Data from years before 2015.
    -   **Validation Set**: Data from the year 2015.
    -   **Test Set**: Data from years after 2015.
-   This time-based split helps in evaluating the model's ability to generalize to future data.

### 4. Logistic Regression Model Training
-   A Logistic Regression model from `sklearn.linear_model` is trained on the preprocessed training data.
-   The `liblinear` solver is used for its efficiency with small datasets and L1/L2 regularization.

### 5. Model Evaluation
-   The model's performance is evaluated using `accuracy_score` and `confusion_matrix` on the training, validation, and test sets.
-   A custom `predict_and_plot` function visualizes the confusion matrix for better interpretability.

### 6. Individual Prediction Function
-   A utility function `predict_input` is provided to make predictions on new, unseen individual data points, applying the same preprocessing steps (imputation, scaling, encoding) as used during training.

### 7. Model Persistence
-   The trained model and all preprocessing objects (imputer, scaler, encoder) are saved using `joblib.dump` into a single file (`aussie_rain.joblib`) for future use and deployment.

## Usage
To run this notebook:
1.  Ensure you have the `weatherAUS.csv` file in the same directory or adjust the path accordingly.
2.  Run all cells sequentially.
3.  You can modify the `new_input` dictionary to test predictions with different weather conditions.
