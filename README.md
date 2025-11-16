📊 Telecom Customer Churn Prediction
Machine Learning + Flask Web Application

This project is an end-to-end Telecom Customer Churn Prediction System, where a machine learning model predicts whether a customer is likely to Churn or Stay, based on demographic, service usage, billing, and account features.
A fully functional Flask web app is included to make real-time predictions through a user-friendly form.

🚀 Features of This Project
✔️ End-to-End ML Workflow

Data cleaning & preprocessing
Feature engineering
Handling missing values
Categorical encoding
Scaling numerical features
Train-test split
Random Forest model training
Model saving using joblib

✔️ Deployment-Ready Flask Backend

Loads saved model + feature metadata

Dynamically generates input fields

Accepts user inputs & returns predictions

Outputs churn status + probability score

✔️ Well-Structured Codebase

save_model.py → Training pipeline

app.py → Web application

features.json → Metadata for dynamic form

model.joblib → Trained model

Jupyter notebook included

📁 Project Structure
Telecom-Churn-Prediction/
│
├── app.py
├── save_model.py
├── requirements.txt
├── model.joblib
├── features.json
├── telecom_customer_churn_prediction.ipynb
├── telecom_customer_churn.csv
│
├── templates/
│   ├── index.html
│   ├── result.html

🧠 Machine Learning Pipeline
1. Data Preparation

Filtered only “Churned” and “Stayed” customers

Created binary target column Churn

Engineered new feature → Revenue_per_Month

Removed unnecessary columns such as Customer ID, Churn Reason, etc.

2. Preprocessing

Missing values handled using SimpleImputer

One-Hot Encoding for categorical features

Standard scaling for numerical features

3. Model Used

Random Forest Classifier

200 trees

random_state=42

Integrated into a Pipeline

📈 Model Evaluation

(Add your metrics from notebook here—for example:)

Accuracy: 0.86

Precision: 0.86

Recall: 0.96

AUC Score: 0.87

You can now enter customer data and get churn predictions in real-time.

🧩 Technologies Used

Python

Pandas / NumPy

Scikit-Learn

Flask

Joblib

HTML / CSS

🔗 Live Demo / GitHub Repo Link

(Add your GitHub link here once uploaded)
