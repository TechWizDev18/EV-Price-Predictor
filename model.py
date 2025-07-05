import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import numpy as np

def load_and_merge_datasets(path1, path2):
    """
    Loads two CSV datasets from the given paths and merges them into a single DataFrame.
    """
    print(f"Loading dataset 1 from: {path1}")
    df1 = pd.read_csv(path1)
    print(f"Dataset 1 shape: {df1.shape}")
    
    print(f"Loading dataset 2 from: {path2}")
    df2 = pd.read_csv(path2)
    print(f"Dataset 2 shape: {df2.shape}")
    
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Merged dataset shape: {merged_df.shape}")
    
    return merged_df

def preprocess_data(df, target_column):
    """
    Preprocesses the DataFrame with enhanced debugging information.
    """
    print(f"\n=== PREPROCESSING DEBUG INFO ===")
    print(f"Original dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    print(f"Columns in dataset: {df.columns.tolist()}")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Check target column before cleaning
    print(f"\nTarget column '{target_column}' sample values before cleaning:")
    print(df[target_column].head(10))
    print(f"Target column data type: {df[target_column].dtype}")
    
    # Drop rows where the target column is missing
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    print(f"Rows dropped due to missing target: {initial_rows - len(df)}")

    # Clean and convert the target column ('PriceinGermany') to float
    df[target_column] = df[target_column].astype(str).str.replace(r'[€,]', '', regex=True)
    
    # Check for any remaining non-numeric characters
    print(f"\nTarget column after cleaning (sample):")
    print(df[target_column].head(10))
    
    # Convert to float
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    
    # Check target statistics
    print(f"\nTarget column statistics:")
    print(f"Min: {df[target_column].min()}")
    print(f"Max: {df[target_column].max()}")
    print(f"Mean: {df[target_column].mean():.2f}")
    print(f"Median: {df[target_column].median():.2f}")
    print(f"NaN count: {df[target_column].isna().sum()}")

    # List of common numeric columns that might contain non-numeric characters
    columns_to_clean_and_convert = [
        'Acceleration', 'TopSpeed', 'Range', 'Efficiency', 'FastCharge',
        'Battery', 'NumberofSeats', 'Power', 'Torque', 'Weight', 'Trunk', 'Seats'
    ]

    print(f"\nCleaning feature columns...")
    for col in columns_to_clean_and_convert:
        if col in df.columns:
            print(f"Processing column: {col}")
            original_values = df[col].head(5)
            
            # Convert column to string first, then clean
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            cleaned_values = df[col].head(5)
            print(f"  Original: {original_values.tolist()}")
            print(f"  Cleaned:  {cleaned_values.tolist()}")
            print(f"  NaN count: {df[col].isna().sum()}")

    # Check data quality before dropping NaNs
    print(f"\nData quality check before dropping NaNs:")
    print(f"Rows with any NaN: {df.isna().any(axis=1).sum()}")
    print(f"Total rows: {len(df)}")

    # After cleaning, drop any rows that now have NaN values in *any* column
    df = df.dropna()
    print(f"Rows remaining after dropping NaNs: {len(df)}")

    if len(df) < 10:
        print("WARNING: Very few rows remaining after cleaning!")
        print("This could cause poor model performance.")

    # Separate features (X) and target (y)
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Select only numeric columns for features
    numeric_columns = X.select_dtypes(include=['number']).columns
    X = X[numeric_columns]
    
    print(f"\nFinal features selected: {X.columns.tolist()}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    for col in X.columns:
        print(f"{col}: min={X[col].min():.2f}, max={X[col].max():.2f}, mean={X[col].mean():.2f}")

    return X, y

def train_model(X_train, y_train):
    """
    Trains a Linear Regression model with debugging information.
    """
    print(f"\n=== TRAINING DEBUG INFO ===")
    print(f"Training set shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Feature scaling info:")
    print(f"Scaler mean: {scaler.mean_}")
    print(f"Scaler scale: {scaler.scale_}")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    
    return model, scaler

def save_model(model, scaler, feature_columns, filename='model.pkl', model_dir='.'):
    """
    Saves the trained model, scaler, and the list of feature columns to a pickle file.
    """
    full_path = os.path.join(model_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'wb') as f:
        # Save model, scaler, and feature_columns as a tuple
        pickle.dump((model, scaler, feature_columns), f)
    print(f"Model, scaler, and feature columns saved as {full_path}")

def load_model(filename='model.pkl', model_dir='.'):
    """
    Loads the model, scaler, and feature columns from a pickle file.
    """
    full_path = os.path.join(model_dir, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file '{full_path}' not found. Please ensure model.py has been run to create it.")
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def main():
    """
    Main function with enhanced debugging and validation.
    """
    # Define paths to your datasets
    path1 = r'C:\Users\dell\Documents\DMA Lab\flask project1\flask project1\Cheapestelectriccars-EVDatabase.csv'
    path2 = r'C:\Users\dell\Documents\DMA Lab\flask project1\flask project1\Cheapestelectriccars-EVDatabase 2023.csv'
    target_column = 'PriceinGermany'

    model_save_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Check if files exist
    if not os.path.exists(path1):
        print(f"ERROR: File not found: {path1}")
        return
    if not os.path.exists(path2):
        print(f"ERROR: File not found: {path2}")
        return
    
    print(f"Loading and merging datasets...")
    df = load_and_merge_datasets(path1, path2)

    print(f"\nPreprocessing data...")
    X, y = preprocess_data(df, target_column)

    # Ensure X is not empty after preprocessing
    if X.empty:
        raise ValueError("No numeric features found after preprocessing. Check your data and target column.")

    # Capture the feature columns
    feature_columns = X.columns.tolist()
    print(f"\nFeatures used for training ({len(feature_columns)}): {feature_columns}")

    print(f"\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTraining model...")
    model, scaler = train_model(X_train, y_train)

    print(f"\nEvaluating model...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.4f}')
    print(f'RMSE: {np.sqrt(mse):.2f}')
    
    # Check prediction range
    print(f"\nPrediction analysis:")
    print(f"Actual prices - Min: €{y_test.min():.2f}, Max: €{y_test.max():.2f}")
    print(f"Predicted prices - Min: €{y_pred.min():.2f}, Max: €{y_pred.max():.2f}")
    
    # Check for negative predictions
    negative_predictions = y_pred < 0
    if negative_predictions.any():
        print(f"WARNING: {negative_predictions.sum()} negative predictions found!")
        print(f"Most negative prediction: €{y_pred.min():.2f}")
    
    # Test with sample data
    print(f"\nTesting with sample data...")
    sample_input = pd.DataFrame([{col: X_test.iloc[0][col] for col in feature_columns}])
    sample_scaled = scaler.transform(sample_input)
    sample_pred = model.predict(sample_scaled)[0]
    sample_actual = y_test.iloc[0]
    
    print(f"Sample prediction: €{sample_pred:.2f}")
    print(f"Sample actual: €{sample_actual:.2f}")
    print(f"Sample input features: {sample_input.iloc[0].to_dict()}")

    # Save the model
    save_model(model, scaler, feature_columns, filename='model.pkl', model_dir=model_save_directory)
    print(f"\nModel training and saving process complete.")

if __name__ == '__main__':
    main()