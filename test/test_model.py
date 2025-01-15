import pandas as pd
import numpy as np
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.preprocess import preprocess_data, scale_numerical_data, create_train_test_split
from dotenv import load_dotenv

    
# Load environment variables
load_dotenv()
mlflow_vars = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
for var in mlflow_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")
    os.environ[var] = os.getenv(var)

class TestPreprocess(unittest.TestCase):
    def test_preprocess_data(self):
        """Test the preprocess_data function."""
        # Load data
        df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")

        # Preprocess the data
        processed_df = preprocess_data(df)
        
        # Make sure the data does not have null values
        self.assertFalse(processed_df.isnull().values.any())

        #Check if the log transform is being applied
        self.assertTrue((processed_df[['Distance','Distance_per_year']] > 0).all().all())
        if 'Price' in df:
            self.assertTrue((processed_df['Price'] > 0).all())
        self.assertNotIn("Year",processed_df.columns)
        self.assertIn("Age",processed_df.columns)

    def test_scale_numerical_data(self):
        """Test the scale_numerical_data function."""
        # Load data
        df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")

        # Preprocess the data
        processed_df = preprocess_data(df)
       
        # Split the data
        X_train, X_test, _, _ = create_train_test_split(processed_df, processed_df['Price'])

        #Scale the data
        X_train_scaled, X_test_scaled = scale_numerical_data(X_train, X_test)

        #Assert not null values in scaled data
        self.assertFalse(X_train_scaled.isnull().values.any())
        self.assertFalse(X_test_scaled.isnull().values.any())
        self.assertNotEqual(X_test.iloc[0][['Age','Distance','Distance_per_year']].values.tolist(),X_test_scaled.iloc[0][['Age','Distance','Distance_per_year']].values.tolist())

if __name__ == '__main__':
    unittest.main()