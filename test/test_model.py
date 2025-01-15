import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import unittest
from src.models.preprocess import preprocess_data, scale_numerical_data, create_train_test_split

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("MLFLOW_TRACKING_URI", type=str, help="MLflow tracking URI")
parser.add_argument("MLFLOW_TRACKING_USERNAME", type=str, help="MLflow tracking username")
parser.add_argument("MLFLOW_TRACKING_PASSWORD", type=str, help="MLflow tracking password")

# Parse the command line arguments
args = parser.parse_args()

#Set the environment variables
os.environ["MLFLOW_TRACKING_URI"] = args.MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_USERNAME"] = args.MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = args.MLFLOW_TRACKING_PASSWORD

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