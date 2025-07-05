import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.preprocessing import StandardScaler # Required for StandardScaler object type
import numpy as np # Ensure numpy is imported

app = Flask(__name__)

# --- Model Loading Logic ---
MODEL_SAVE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = 'model.pkl'
MODEL_FULL_PATH = os.path.join(MODEL_SAVE_DIRECTORY, MODEL_FILENAME)

model = None
scaler = None
expected_features = None # To store the feature columns loaded from the model file

def load_trained_model():
    """
    Loads the trained model, scaler, and the list of expected features from the
    'model.pkl' file. This function is called once when the Flask app starts.
    """
    global model, scaler, expected_features
    try:
        if not os.path.exists(MODEL_FULL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_FULL_PATH}' not found. Please ensure model.py has been run successfully to create it.")
        
        # Load the model, scaler, AND expected_features as a tuple
        with open(MODEL_FULL_PATH, 'rb') as f:
            model, scaler, expected_features = pickle.load(f)
        
        print(f"Successfully loaded model, scaler, and features from: {MODEL_FULL_PATH}")
        print(f"Model expects {len(expected_features)} features: {expected_features}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Model could not be loaded. Predictions will not be available.")
    except Exception as e:
        # Catch any other unexpected errors during loading
        print(f"An unexpected error occurred while loading the model: {e}")
        print("Model could not be loaded. Predictions will not be available.")

# Call the model loading function when the application starts
load_trained_model()

# --- Flask Routes and Prediction Logic ---

@app.route('/')
def home():
    """
    Renders the home page of the application.
    Assumes 'index.html' is located in a 'templates' subfolder.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests with proper error handling and feature validation.
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None or expected_features is None:
            return jsonify({'error': 'Model not loaded. Please ensure model.py has been run successfully.'}), 500
        
        # Get JSON data from the request
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({'error': 'No input data received. Please provide JSON data.'}), 400

        print(f"Received data for prediction: {json_data}")

        # Create a pandas DataFrame from the input JSON data
        input_df = pd.DataFrame([json_data])
        
        # Check which expected features are missing from input
        missing_features = [f for f in expected_features if f not in input_df.columns]
        
        # Define reasonable default values based on your actual dataset
        feature_defaults = {
            'Acceleration': 7.0,          # seconds (0-100 km/h)
            'TopSpeed': 180.0,            # km/h
            'Range': 350.0,               # km
            'Efficiency': 200.0,          # Wh/km
            'NumberofSeats': 5.0,         # typical car
            'FastChargeSpeed': 500.0      # km/h (optional feature)
        }
        
        # Add missing features with reasonable defaults
        for feature in missing_features:
            default_value = feature_defaults.get(feature, 0.0)
            input_df[feature] = default_value
            print(f"Warning: Missing feature '{feature}' filled with default value {default_value}")
        
        # Reorder columns to match expected_features order
        input_df = input_df[expected_features]
        
        # Convert all columns to numeric, handling any non-numeric values
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Fill any NaN values that resulted from conversion with 0
        input_df = input_df.fillna(0)
        
        print(f"Processed input data shape: {input_df.shape}")
        print(f"Input data:\n{input_df}")
        
        # Validate that we have the correct number of features
        if input_df.shape[1] != len(expected_features):
            return jsonify({
                'error': f'Input data mismatch. Expected {len(expected_features)} features '
                         f'but received {input_df.shape[1]} after preprocessing.'
            }), 400
        
        # Scale the input data using the loaded scaler
        features_scaled = scaler.transform(input_df)
        
        # Make the prediction using the loaded model
        prediction = model.predict(features_scaled)[0]
        
        # Return the prediction as a JSON response
        return jsonify({
            'predicted_price': float(prediction),
            'features_used': expected_features,
            'missing_features_filled': missing_features
        })

    except Exception as e:
        # Catch any other general errors during the prediction process
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

@app.route('/features', methods=['GET'])
def get_expected_features():
    """
    Returns the list of expected features for the model.
    Useful for debugging and API documentation.
    """
    if expected_features is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'expected_features': expected_features,
        'feature_count': len(expected_features)
    })

# Run the Flask application in debug mode
if __name__ == '__main__':
    app.run(debug=True)