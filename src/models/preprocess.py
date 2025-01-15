import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, is_prediction=False):
    """Preprocesses the used car dataset with improved scaling and transformations."""
    df = df.copy()

    # Handle missing values consistently
    for col in ['Car Name', 'Location']:
        df[col] = df[col].fillna("Unknown")


    # Handle Price only if it's not a prediction case
    if not is_prediction and 'Price' in df.columns:
        df['Price'] = np.log10(df['Price'])
    elif is_prediction and 'Price' in df.columns:
        df = df.drop('Price', axis=1)

    # Fill missing values
    df["Car Name"] = df["Car Name"].fillna("Unknown")
    df["Year"] = df["Year"].fillna(df["Year"].mode()[0])
    df["Location"] = df["Location"].fillna("Unknown")

    # Convert Year to integer
    df["Year"] = df["Year"].fillna(df["Year"].mode()[0]).astype(int)
    
    # Create relative age feature
    current_year = 2025
    df['Age'] = current_year - df['Year']
    df.drop('Year', axis=1, inplace=True)

    # Feature engineering for Distance
    df['Distance_per_year'] = df['Distance'] / (df['Age'] + 1)

    # Scale transformation for numerical features (using log base 10 for better scale)
    for col in ['Distance', 'Distance_per_year']:
        # Add 1 to handle zero values
        df[col] = np.log10(df[col] + 1)

    # Scale transformation for Price (target variable)
    #if 'Price' in df:
    # df['Price'] = np.log10(df['Price'])

    return df

def create_train_test_split(df, target_series, test_size=0.2, random_state=42):
    """Splits the preprocessed dataframe into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        df, target_series, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def scale_numerical_data(X_train, X_test):
    """Scales only the numerical features using RobustScaler."""
    numerical_cols = ['Age', 'Distance', 'Distance_per_year']
    scaler = RobustScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_test_scaled

def inverse_transform_price(y_pred):
    """Inverse transforms the log-transformed price predictions."""
    return np.power(10, y_pred)

if __name__ == '__main__':
    # Load data
    df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")
    
    # Preprocess the data
    processed_df = preprocess_data(df)
    
    # Separate target variable (already log-transformed)
    target = np.log10(df['Price'])
    
    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(processed_df, target)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_numerical_data(X_train, X_test)

    print("Preprocessed data shape:", X_train_scaled.shape)
    print("\nSample of preprocessed features:")
    print(X_train_scaled.head())