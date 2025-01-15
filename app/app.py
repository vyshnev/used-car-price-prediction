import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.preprocess import preprocess_data, scale_numerical_data, inverse_transform_price
from sklearn.preprocessing import OneHotEncoder
import boto3
import io
import dvc.api

# Load environment variables
load_dotenv()
mlflow_vars = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
for var in mlflow_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")
    os.environ[var] = os.getenv(var)

os.environ["AWS_S3_BUCKET_NAME"] = os.getenv("AWS_S3_BUCKET_NAME")

def load_latest_model():
    """Loads the latest model from MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_name = "Used Car Price Prediction"
        experiments = client.search_experiments(filter_string=f"name = '{experiment_name}'")
        
        if not experiments:
            raise ValueError(f"No experiment found with the name '{experiment_name}'")
            
        experiment_id = experiments[0].experiment_id
        runs = client.search_runs(experiment_ids=[experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
        
        if not runs:
            raise ValueError(f"No runs found for the experiment '{experiment_name}'")
            
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        model = mlflow.sklearn.load_model(model_uri)
        return model
        
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

def load_data_from_s3():
    """Loads the processed data from S3."""
    try:
        path = 'data/raw/cars24-used-cars-dataset.csv'
        bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        repo = '.'
        with dvc.api.open(
            path=path,
            repo=repo,
            mode='r',
        ) as fd:
            df = pd.read_csv(fd)
        return df
    except Exception as e:
        st.error(f"Error loading the data from S3: {e}")
        return None

# Load data
df = load_data_from_s3()
if df is None:
    st.stop()

# Define all possible values
all_car_names = df['Car Name'].unique().tolist()
all_fuels = df['Fuel'].unique().tolist()
all_locations = df['Location'].unique().tolist()
all_drives = df['Drive'].unique().tolist()
all_types = df['Type'].unique().tolist()
all_owners = df['Owner'].unique().tolist()

# Define features to one-hot encode
categorical_features = ['Car Name', 'Fuel', 'Location', 'Drive', 'Type', 'Owner']

st.title('Used Car Price Prediction')

# Create input widgets
st.sidebar.header("Enter Car Details")
car_name = st.sidebar.text_input("Car Name", "Tata Tiago")
distance = st.sidebar.number_input("Distance (km)", min_value=0, value=50000)
owner = st.sidebar.selectbox("Owner", [1, 2, 3, 4])
fuel = st.sidebar.selectbox("Fuel Type", ["PETROL", "DIESEL", "CNG", "ELECTRIC", "LPG"])
location = st.sidebar.text_input("Location (e.g., HR-26)", "HR-99")
gear_type = st.sidebar.selectbox("Drive", ["Manual", "Automatic"])
car_type = st.sidebar.selectbox("Type", ["HatchBack", "Sedan", "SUV", "MUV", "Convertible"])
age = st.sidebar.number_input("Age", min_value=0, max_value=20, value=5)
current_year = 2025
year = current_year - age

if st.sidebar.button('Predict Price'):
    try:
        # Load model first to check expected features
        model = load_latest_model()
        if model is not None:
            expected_features = model.feature_names_in_.tolist()
            st.write("Model's expected features:", expected_features)
        
        # Prepare input data
        input_data = {
            "Car Name": car_name,
            "Distance": distance,
            "Owner": int(owner),
            "Fuel": fuel,
            "Location": location,
            "Drive": gear_type,
            "Type": car_type,
            "Age": age,
            "Year": year
        }
        
        # Convert to a dataframe
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the data
        processed_df = preprocess_data(input_df, is_prediction=True)
        
        # Create a DataFrame with all expected features initialized to 0
        final_df = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Handle numerical features first
        numerical_cols = ['Age', 'Distance', 'Distance_per_year']
        X_scaled = scale_numerical_data(
            processed_df[numerical_cols].copy(),
            processed_df[numerical_cols].copy()
        )[0]
        
        # Add scaled numerical features
        for col in numerical_cols:
            final_df[col] = X_scaled[col].values
        
        # Handle categorical features
        categorical_mappings = {
            'Car Name': f"Car Name_{input_data['Car Name']}",
            'Fuel': f"Fuel_{input_data['Fuel']}",
            'Location': f"Location_{input_data['Location']}",
            'Drive': f"Drive_{input_data['Drive']}",
            'Type': f"Type_{input_data['Type']}",
            'Owner': f"Owner_{str(input_data['Owner'])}"
        }
        
        # Set categorical features
        for feature, encoded_feature in categorical_mappings.items():
            if encoded_feature in expected_features:
                final_df[encoded_feature] = 1
            else:
                # If the exact feature doesn't exist, use the Unknown version
                unknown_feature = f"{feature}_Unknown"
                if unknown_feature in expected_features:
                    final_df[unknown_feature] = 1
        
        # Verify the features
        st.write("Features we're providing:", final_df.columns.tolist())
        st.write("Features missing from our input:", 
                set(model.feature_names_in_) - set(final_df.columns))
        st.write("Extra features in our input:", 
                set(final_df.columns) - set(model.feature_names_in_))
        
        # Make prediction
        prediction = model.predict(final_df)
        prediction = inverse_transform_price(prediction)
        st.success(f"Predicted Price: {prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Debug info:")
        if 'model' in locals() and hasattr(model, 'feature_names_in_'):
            st.write("Expected features:", model.feature_names_in_.tolist())
        if 'final_df' in locals():
            st.write("Provided features:", final_df.columns.tolist())
        st.write("Input data shape:", input_df.shape if 'input_df' in locals() else "Not created")
        if 'processed_df' in locals():
            st.write("Processed data columns:", processed_df.columns.tolist())
            st.write("Processed data sample:", processed_df.head())