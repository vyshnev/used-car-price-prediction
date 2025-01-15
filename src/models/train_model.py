import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
from preprocess import preprocess_data, scale_numerical_data, inverse_transform_price
import os
from dotenv import load_dotenv
from typing import Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Get MLflow tracking variables from environment
mlflow_vars = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]
for var in mlflow_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing required environment variable: {var}")
    os.environ[var] = os.getenv(var)


def calculate_metrics(model: Any, X_train_scaled: pd.DataFrame,
                     X_test_scaled: pd.DataFrame, y_train: pd.Series,
                     y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics for the model."""
    try:
        # Convert predictions and actual values back to original scale
        y_test_original = inverse_transform_price(y_test)
        y_pred_original = inverse_transform_price(y_pred)

        # Handle any potential infinity values
        mask = np.isfinite(y_pred_original) & np.isfinite(y_test_original)
        y_test_original = y_test_original[mask]
        y_pred_original = y_pred_original[mask]

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_original, y_pred_original))),
            "r2": float(r2_score(y_test_original, y_pred_original))
        }

        # Perform cross-validation if applicable
        if not isinstance(model, (KNeighborsRegressor, XGBRegressor)):
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            metrics.update({
                "mean_cv_rmse": float(cv_rmse.mean()),
                "std_cv_rmse": float(cv_rmse.std())
            })

        return metrics
    except Exception as e:
        logging.error(f"Error in calculate_metrics: {str(e)}")
        raise


def train_and_evaluate_model(
    model: Any,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    param_grid: Dict = None,
    search_type: str = "grid"
) -> Dict[str, Any]:
    """Trains and evaluates a given regression model with optional hyperparameter tuning."""
    experiment_name = "Used Car Price Prediction"
    logging.info(f"Training {model.__class__.__name__} model")

    try:
        # Set the MLflow experiment
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            best_params = {}

            # Hyperparameter tuning (if a parameter grid is provided)
            if param_grid:
                logging.info(f"Performing hyperparameter tuning for {model.__class__.__name__}")
                # Use GridSearchCV or RandomizedSearchCV based on the search_type variable
                search = (GridSearchCV if search_type == "grid" else RandomizedSearchCV)(
                    model, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1
                )
                search.fit(X_train_scaled, y_train) # Perform hyperparameter search
                model = search.best_estimator_ # Set the model to the best estimator found
                best_params = search.best_params_  # Save the best parameters
                logging.info(f"Best parameters for {model.__class__.__name__}: {best_params}")

            # Train the model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            metrics = calculate_metrics(model, X_train_scaled, X_test_scaled,
                                     y_train, y_test, y_pred)

            # Log parameters and metrics to MLflow
            mlflow.log_params(best_params or model.get_params()) # Log best parameters, or all parameters if no tuning
            mlflow.log_metrics(metrics) # Log the calculated metrics

            # Log model (save it to MLflow)
            mlflow.sklearn.log_model(model, "model")

            logging.info(f"Model evaluation metrics: {metrics}")
            # Return a dictionary with the model and its metrics
            return {"model": model, "best_params": best_params, "metrics": metrics, "run_id": run.info.run_id}

    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise


def main():
    """Main function to load, preprocess, train, and evaluate models."""
    try:
        # Load data
        logging.info("Loading and preprocessing data...")
        df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")

        # Preprocess data and get log-transformed price
        processed_df = preprocess_data(df)
        target = np.log10(df['Price'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(processed_df, target, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled, X_test_scaled = scale_numerical_data(X_train, X_test)

        # Define models and their parameter grids
        models = {
            "Linear Regression": (
                LinearRegression(),
                {}
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            ),
            "Gradient Boosting": (
                GradientBoostingRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            ),
            "Decision Tree": (
                DecisionTreeRegressor(random_state=42),
                {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                }
            ),
            "K-Nearest Neighbors": (
                KNeighborsRegressor(),
                {"n_neighbors": [3, 5, 10]}
            ),
            "XGBoost": (
                XGBRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7]
                }
            )
        }

        # Train and evaluate models
        model_metrics = {}
        for model_name, (model, param_grid) in models.items():
            logging.info(f"Training {model_name}...")
            try:
                results = train_and_evaluate_model(
                    model, X_train_scaled, X_test_scaled,
                    y_train, y_test, param_grid
                )
                model_metrics[model_name] = results
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue
        print("Features used during training:", X_train.columns.tolist())
        # Save this list somewhere for reference
        # Print results
        print("\nModel Training Results:")
        print("=" * 50)
        for model_name, results in model_metrics.items():
            print(f"\nModel: {model_name}")
            metrics = results['metrics']
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  R² Score: {metrics['r2']:.4f}")
            if 'mean_cv_rmse' in metrics:
                print(f"  Mean CV RMSE: {metrics['mean_cv_rmse']:.2f}")
                print(f"  Std CV RMSE: {metrics['std_cv_rmse']:.2f}")
            print(f"  Best Parameters: {results['best_params']}")
            print(f"  MLflow Run ID: {results['run_id']}")
            print("-" * 50)
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == '__main__':
    main()