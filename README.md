Financial Distress Prediction Application
This project is a web-based application designed to predict the financial distress of companies using machine learning models. The app allows users to upload their dataset, select relevant columns, and visualize predictions regarding whether companies are financially distressed.

Table of Contents
Project Overview
Dataset
Installation
Running the Application
Features
Model Details
License
Project Overview
The primary goal of this project is to predict financial distress in companies using a predictive model developed with Python and deployed using Streamlit.

Key Highlights:
Utilizes Logistic Regression, Random Forest, and XGBoost for predictions.
Includes Exploratory Data Analysis (EDA) with visualizations like heatmaps and boxplots.
Encodes categorical data and scales numerical features.
Displays prediction results, including the percentage of distressed and healthy companies.
Dataset
The dataset contains financial information for companies over different periods, with features related to their financial health. Key columns include:

Financial Distress: Binary target column where companies are classified as either distressed (1) or healthy (0).
x80 (Categorical Column): One-hot encoded categorical data.
The dataset does not contain any missing values.

Installation
Prerequisites:
Python 3.8+
Libraries: streamlit, pandas, joblib, matplotlib, sklearn, xgboost
Clone the repository:

bash
Copy code
git clone <your-repo-url>
Navigate to the project directory:

bash
Copy code
cd financial-distress-prediction
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Start the Streamlit app:

bash
Copy code
streamlit run app.py
Running the Application
Upload your CSV file containing company financial data.
Select the relevant columns:
Financial Distress Column: The column indicating whether a company is distressed.
Categorical Column (x80): The column that needs encoding.
View the predictions and visualizations, including a bar chart displaying the number of distressed and healthy companies.
Features
CSV File Upload: Upload your custom dataset to the application.
Column Mapping: Select columns for financial distress and categorical values manually.
Prediction and Visualization: View the predicted financial status of companies and visualizations of the results.
Matplotlib Integration: Bar chart of the number of distressed vs. healthy companies.
Model Details
The application uses three machine learning models:

Logistic Regression
Random Forest
XGBoost (best-performing model)
The XGBoost model is trained on an imbalanced dataset using SMOTE for resampling. Feature scaling is applied using StandardScaler.

Evaluation metrics include:

F1-Score (used due to imbalanced nature of the dataset)
Accuracy and AUC for performance comparisons
License
