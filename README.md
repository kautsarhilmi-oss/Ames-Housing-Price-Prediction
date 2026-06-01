# Ames House Price Prediction - Advanced Regression Ensemble

This repository contains an end-to-end Machine Learning pipeline developed for the classic Kaggle competition: **"House Prices - Advanced Regression Techniques"**. The goal of this project is to predict the final sales price of residential homes in Ames, Iowa, using 79 explanatory variables.

## 📊 Project Overview
Predicting house prices requires careful handling of both high-dimensional categorical features and highly skewed numerical distributions. This solution leverages advanced feature engineering and a robust multi-model ensemble strategy to minimize the Root-Mean-Squared-Error (RMSE) between the log of the predicted value and the log of the actual sales price.

## ⚙️ Core Pipeline Steps

### 1. Data Cleaning & Preprocessing
* **Missing Value Imputation**: Handled missing values based on data context (e.g., filling missing garage variables with 'No Garage' or numerical missing values with the median/mode of the neighborhood).
* **Target Transformation**: Applied Log Transformation (`log1p`) to the target variable `SalePrice` to fix right-skewness and meet the normality assumptions of regression models.

### 2. Creative Feature Engineering
* **Feature Synthesis**: Combined multiple structural columns to create comprehensive metrics (e.g., `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`).
* **Property Age Analysis**: Engineered `HouseAge` and `YearsSinceRemod` using `YrSold`, `YearBuilt`, and `YearRemodAdd`.
* **Encoding**: Implemented robust encoding techniques for categorical variables to capture ordinal relationships accurately.

### 3. Model Architecture & Ensemble Strategy
To avoid overfitting and maximize predictive power, the pipeline implements an ensemble strategy validated via **K-Fold Cross-Validation**:
* **Ridge Regression**: Captures linear relationships with L2 regularization.
* **Lasso & ElasticNet**: Performs feature selection and continuous shrinkage.
* **Gradient Boosting (XGBoost / LightGBM)**: Progressively minimizes residuals for complex patterns.
* **Weighted Average**: Combines predictions based on out-of-fold validation scores to produce the final `submission.csv`.

## 🚀 How to Run
1. Clone this repository.
2. Place `train.csv` and `test.csv` in the root directory.
3. Execute the pipeline notebook or script.
