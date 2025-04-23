🧠 Employee Attrition Prediction using Machine Learning
This repository contains a machine learning pipeline built to predict employee attrition (turnover) using HR analytics data. The solution applies data preprocessing, visualization, and multiple classification models to assess the likelihood of employees leaving the company based on various workplace and demographic factors.

📊 Project Overview
Employee attrition is a critical issue in organizations, affecting productivity and continuity. This project explores various features such as:

Job Satisfaction

Performance Metrics

Tenure

Monthly Income

Work-Life Balance

Overtime

And more...

The goal is to accurately classify whether an employee is likely to leave the organization using historical data.

🛠️ Technologies Used
Python 3

Pandas & NumPy

Matplotlib & Seaborn

Scikit-learn

XGBoost

imbalanced-learn (SMOTE)

Joblib

🔍 Key Features
✅ Data Cleaning & Preprocessing
Dropping irrelevant features

Encoding categorical values

Feature scaling

Balancing the dataset using SMOTE

📈 Exploratory Data Analysis (EDA)
Distribution and relationship plots for key features

Crosstabs and heatmaps for insight discovery

🧪 Model Training & Evaluation
Models used:

Logistic Regression

Random Forest

XGBoost

Evaluation metrics:

Accuracy

ROC-AUC

Classification Report

🔧 Hyperparameter Tuning
Logistic Regression tuned using GridSearchCV

🔎 Feature Importance Analysis
Extraction and visualization of top features influencing attrition

💾 Model Persistence
Saving trained model, scaler, and prediction outputs for reuse                                                         

Dash App: https://employee-attrition-prediction-9cwl.onrender.com
