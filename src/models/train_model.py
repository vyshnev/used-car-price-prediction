import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
from preprocess import preprocess_data, create_train_test_split, scale_numerical_data
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

def train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Trains and evaluates a given regression model.

    Args:
        model: A scikit-learn regression model
        X_train_scaled (pd.DataFrame): Scaled training features
        X_test_scaled (pd.DataFrame): Scaled testing features
        y_train (pd.Series): Training target variable
        y_test (pd.Series): Testing target variable

    Returns:
         dict: Dictionary containing the model and its metrics
    """
    # Initialize MLflow experiment
    mlflow.set_experiment("Used Car Price Prediction")

    with mlflow.start_run():
      # Train the model
      model.fit(X_train_scaled, y_train)

      # Make predictions
      y_pred = model.predict(X_test_scaled)

      # Calculate evaluation metrics
      rmse = np.sqrt(mean_squared_error(y_test, y_pred))
      r2 = r2_score(y_test, y_pred)
      cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
      cv_rmse = np.sqrt(-cv_scores)
      mean_cv_rmse = cv_rmse.mean()
      std_cv_rmse = cv_rmse.std()

      # Log parameters and metrics to MLflow
      mlflow.log_metric("rmse", rmse)
      mlflow.log_metric("r2", r2)
      mlflow.log_metric("mean_cv_rmse", mean_cv_rmse)
      mlflow.log_metric("std_cv_rmse", std_cv_rmse)

      # Log the trained model
      mlflow.sklearn.log_model(model, "model")

    return {"model": model, "metrics": {"rmse": rmse, "r2": r2, "mean_cv_rmse": mean_cv_rmse, "std_cv_rmse": std_cv_rmse}}

if __name__ == '__main__':
    # Load data
    df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")

    # Preprocess the data
    df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(df)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_numerical_data(X_train, X_test)

    # Define the models to evaluate
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(random_state=42)
    }

    # Train and evaluate the models
    model_metrics = {}
    for model_name, model in models.items():
        results = train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        model_metrics[model_name] = results

    print("Model Training Completed.\n")
    for model_name, results in model_metrics.items():
      print(f"Model: {model_name}")
      print(f"  RMSE: {results['metrics']['rmse']:.4f}")
      print(f"  R2 Score: {results['metrics']['r2']:.4f}")
      print(f"  Mean CV RMSE: {results['metrics']['mean_cv_rmse']:.4f}")
      print(f"  Std CV RMSE: {results['metrics']['std_cv_rmse']:.4f}")
      print("-" * 40)