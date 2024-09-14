import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to show instructions in a popup window
def show_instructions():
    st.sidebar.title("Instructions")
    st.sidebar.write("""
        - **Upload your CSV file**: This should contain the financial data of companies.
        - **Select the correct columns**:
            - **Financial Distress Column**: The column that indicates if a company is financially distressed or not.
            - **Categorical Column (x80)**: The column that contains categorical data which needs to be encoded.
        - **Model Predictions**: The app will display predictions, counts, and a bar chart of the predicted financial distress.
    """)

# Show instructions on the sidebar
show_instructions()

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display first few rows
    st.write("Here is a preview of your uploaded data:")
    st.dataframe(df.head())
    
    # Allow users to select the columns manually
    st.write("Please map the columns correctly for the model to work.")
    
    financial_distress_col = st.selectbox("Select the 'Financial Distress' column:", df.columns)
    categorical_col = st.selectbox("Select the categorical column (x80):", df.columns)
    
    # Ensure the selected columns are correctly displayed
    st.write(f"Selected 'Financial Distress' column: {financial_distress_col}")
    st.write(f"Selected categorical column (x80): {categorical_col}")
    
    # Proceed with encoding categorical variables
    df_encoded = pd.get_dummies(df, columns=[categorical_col], drop_first=True)
    
    st.write("Encoded Data Preview:")
    st.dataframe(df_encoded.head())
    
    # Load the pre-trained XGBoost model
    model = joblib.load("xgboost_model.pkl")  # Adjust to your model filename and location
    
    # Separate features and target variable
    X = df_encoded.drop([financial_distress_col], axis=1)
    y = df_encoded[financial_distress_col]
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Ensure you use the same scaling as during training
    
    # Make predictions using the model
    predictions = model.predict(X_scaled)
    
    # Add predictions to the DataFrame
    df_encoded['Predicted_Financial_Distress'] = predictions
    
    # Display the prediction counts (how many 0s and 1s)
    st.write("Predictions (Distribution of Distressed vs. Healthy Companies):")
    st.dataframe(df_encoded['Predicted_Financial_Distress'].value_counts())
    
    # Count total distressed companies
    total_distressed = (df_encoded['Predicted_Financial_Distress'] == 1).sum()
    total_healthy = (df_encoded['Predicted_Financial_Distress'] == 0).sum()
    
    # Display the total count of distressed and healthy companies
    st.write(f"Total Distressed Companies: {total_distressed}")
    st.write(f"Total Healthy Companies: {total_healthy}")
    
    # Optionally, display the percentage of distressed companies
    percentage_distressed = (total_distressed / len(df_encoded)) * 100
    st.write(f"Percentage of Distressed Companies: {percentage_distressed:.2f}%")
    
    # Matplotlib Bar Chart
    st.write("Bar Chart of Predictions:")
    fig, ax = plt.subplots()
    ax.bar(['Distressed', 'Healthy'], [total_distressed, total_healthy], color=['red', 'green'])
    ax.set_xlabel('Company Status')
    ax.set_ylabel('Number of Companies')
    ax.set_title('Distribution of Predicted Financial Distress')
    st.pyplot(fig)
