from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the scaler and models
scaler = joblib.load('scaler.pkl')
models = {
    'Random Forest': joblib.load('random_forest_model.pkl'),
    'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
    'Neural Network': joblib.load('neural_network_model.pkl'),
    'Ensemble (RF+GB)': joblib.load('ensemble_(rf+gb)_model.pkl')
}

# Define feature columns
FEATURES = ['HOUR', 'MINUTE', 'DAY_OF_WEEK', 'DISTANCE_KM', 'CALL_TYPE_NUM', 
            'RUSH_HOUR', 'DISTANCE_KM_SQ', 'IS_WEEKEND', 'MINUTE_OF_DAY',
            'DISTANCE_RUSH_INTERACTION', 'TRAFFIC_INTENSITY']

# Helper function to preprocess input data
def preprocess_input(data):
    # Extract and parse time
    time_str = data['time']
    hour, minute = map(int, time_str.split(':'))
    
    # Extract other inputs
    distance_km = float(data['distance_km'])
    call_type = data['call_type']
    day_of_week = int(data['day_of_week'])
    
    # Engineer features
    call_type_num = {'A': 0, 'B': 1, 'C': 2}.get(call_type, 1)
    rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
    distance_km_sq = distance_km ** 2
    is_weekend = 1 if day_of_week >= 5 else 0
    minute_of_day = hour * 60 + minute
    distance_rush_interaction = distance_km * rush_hour
    traffic_intensity = (2 if (is_weekend == 0 and (7 <= hour <= 9 or 17 <= hour <= 19)) else
                        1 if (is_weekend == 0 and 9 < hour < 17) else 0)
    
    # Create feature array
    features = [
        hour, minute, day_of_week, distance_km, call_type_num,
        rush_hour, distance_km_sq, is_weekend, minute_of_day,
        distance_rush_interaction, traffic_intensity
    ]
    
    # Scale features
    features_scaled = scaler.transform([features])
    return features_scaled, {
        'Day of Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
        'Time': time_str,
        'Distance (km)': distance_km,
        'Call Type': call_type
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate inputs
        required_fields = ['model', 'input_method', 'time', 'distance_km', 'call_type', 'day_of_week']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing or empty field: {field}'}), 400
        
        # Validate model selection
        if data['model'] not in models:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        # Validate time format
        if len(data['time'].split(':')) != 2:
            return jsonify({'error': 'Invalid time format. Use HH:MM (24-hour)'}), 400
        
        # Validate distance
        try:
            distance_km = float(data['distance_km'])
            if distance_km <= 0 or distance_km > 1000:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'Distance must be a number between 0 and 1000 km'}), 400
        
        # Validate call type
        if data['call_type'] not in ['A', 'B', 'C']:
            return jsonify({'error': 'Call Type must be A, B, or C'}), 400
        
        # Validate day of week
        try:
            day_of_week = int(data['day_of_week'])
            if day_of_week < 0 or day_of_week > 6:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'Day of Week must be a number between 0 (Monday) and 6 (Sunday)'}), 400
        
        # Preprocess input data
        features_scaled, user_inputs = preprocess_input(data)
        
        # Make prediction (convert hours to minutes)
        selected_model = models[data['model']]
        prediction_hours = selected_model.predict(features_scaled)[0]
        prediction_minutes = prediction_hours * 60
        
        return jsonify({
            'prediction': prediction_minutes,
            'user_inputs': user_inputs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)