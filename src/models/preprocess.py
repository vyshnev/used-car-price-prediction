import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    """Preprocesses the used car dataset.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Drop the unnecessary index column
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Fill missing values
    df["Car Name"] = df["Car Name"].fillna("Unknown")
    df["Year"] = df["Year"].fillna(df["Year"].mode()[0])
    df["Location"] = df["Location"].fillna("Unknown")

    # Convert Year to integer
    df["Year"] = df["Year"].astype(int)

     # Outlier handling (using IQR method)
    for col in ["Distance", "Price"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # Log transformation for skewed features
    for col in ["Distance", "Price"]:
        df[col] = np.log1p(df[col])

    # Define features to one-hot encode
    categorical_features = ['Car Name', 'Fuel', 'Location', 'Drive', 'Type', 'Owner']

    # One Hot Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    df.drop(columns=categorical_features, inplace=True)

    return df

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Splits the preprocessed dataframe into train and test sets
    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Controls the shuffling applied to the data before applying the split
    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test
    """
    X = df.drop('Price', axis=1)  # Features
    y = df['Price']               # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def scale_numerical_data(X_train, X_test):
    """Scales the numerical data

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features

    Returns:
        tuple: A tuple containing X_train_scaled and X_test_scaled
    """
    numerical_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train_scaled, X_test_scaled

if __name__ == '__main__':
    # Load data
    df = pd.read_csv("data/raw/cars24-used-cars-dataset.csv")

    # Preprocess the data
    df = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(df)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_numerical_data(X_train, X_test)

    print("Preprocessed data:")
    print(X_train_scaled.head())
    print("Preprocessed data shape: ", X_train_scaled.shape)