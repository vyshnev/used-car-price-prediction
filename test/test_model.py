import pandas as pd
import numpy as np
import unittest
import os
from src.models.preprocess import preprocess_data, scale_numerical_data, inverse_transform_price
from src.etl.data_scraping import load_from_kaggle

class TestDataPreprocessing(unittest.TestCase):
    """Test the functionality of preprocessing data."""

    def setUp(self):
        """Setup method to load a sample dataset and use it for testing purposes."""
        self.df = load_from_kaggle()

    def test_preprocess_data_shape(self):
        """Test that the preprocess_data function does not alter the dataset's row count."""
        processed_df = preprocess_data(self.df)
        self.assertEqual(len(self.df), len(processed_df), "Number of rows is not maintained after preprocessing.")

    def test_preprocess_data_columns(self):
      """Test that preprocess_data correctly adds and transforms numerical columns and do not remove categorical columns"""
      processed_df = preprocess_data(self.df)
      expected_columns = set(['Car Name', 'Distance', 'Owner', 'Fuel', 'Location', 'Gear Type', 'Price','Age','Distance_per_year'])
      self.assertTrue(expected_columns.issubset(processed_df.columns), "The preprocessing step did not keep all the necessary columns.")

    def test_scale_numerical_data_numerical_columns(self):
      """Test that scale_numerical_data correctly scales numerical columns and does not transform the others"""
      processed_df = preprocess_data(self.df)
      X = processed_df.copy()
      numerical_cols = ['Age', 'Distance', 'Distance_per_year']
      X_train_scaled, X_test_scaled = scale_numerical_data(X, X)
      self.assertNotEqual(X[numerical_cols].values.tolist(), X_train_scaled[numerical_cols].values.tolist(), "The numerical scaling did not correctly modify the values.")

    def test_inverse_transform_price(self):
        """Tests that the inverse transformation works correctly."""
        y_pred_log = pd.Series([np.log10(1000), np.log10(10000), np.log10(100000)])
        y_pred_original = inverse_transform_price(y_pred_log)
        expected_original = [1000, 10000, 100000]

        np.testing.assert_allclose(y_pred_original, expected_original, atol=1e-6, err_msg="Error in inverse transforming price")
    
if __name__ == '__main__':
    unittest.main()