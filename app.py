from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np
import pandas as pd
import logging
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Enable CORS for all routes, allowing requests from your GitHub Pages origin
CORS(app, resources={r"/predict": {"origins": "https://alexyip712.github.io"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load models and scaler
try:
    models = {
        'Linear Regression': joblib.load('linear_regression_model.pkl'),
        'Ridge Regression': joblib.load('ridge_regression_model.pkl'),
        'Decision Tree': joblib.load('decision_tree_model.pkl'),
        'Random Forest': joblib.load('random_forest_model.pkl'),
        'Gradient Boosting': joblib.load('gradient_boosting_model.pkl'),
        'XGBoost': joblib.load('xgboost_model.pkl')
    }
    scaler = joblib.load('scaler.pkl')
    logger.info("Models and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models or scaler: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        selected_model = data['model']
        input_method = data['input_method']
        
        # Check if the selected model exists
        if selected_model not in models:
            return jsonify({'error': f"Model '{selected_model}' is not available."}), 400
        
        # Feature columns
        feature_columns = ['HOUR', 'MINUTE', 'DAY_OF_WEEK', 'DISTANCE_KM', 'CALL_TYPE_NUM', 
                          'RUSH_HOUR', 'DISTANCE_KM_SQ', 'IS_WEEKEND', 'MINUTE_OF_DAY',
                          'DISTANCE_RUSH_INTERACTION', 'TRAFFIC_INTENSITY']
        
        if input_method == 'datetime':
            hour, minute = map(int, data['time'].split(':'))
            distance_km = float(data['distance_km'])
            call_type = data['call_type']
            day_of_week = int(data['day_of_week'])
        else:  # map
            hour, minute = map(int, data['time'].split(':'))
            distance_km = float(data['distance_km'])
            call_type = data['call_type']
            day_of_week = int(data['day_of_week'])
        
        # Add distance limit (1000 km)
        if distance_km > 1000:
            return jsonify({'error': 'Distance cannot exceed 1000 km.'}), 400
        
        # Engineer features
        rush_hour = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        distance_km_sq = distance_km ** 2
        call_type_num = {'A': 0, 'B': 1, 'C': 2}[call_type]
        is_weekend = 1 if day_of_week >= 5 else 0
        minute_of_day = hour * 60 + minute
        distance_rush_interaction = distance_km * rush_hour
        traffic_intensity = 2 if (is_weekend == 0 and (7 <= hour <= 9 or 17 <= hour <= 19)) else \
                           1 if (is_weekend == 0 and 9 < hour < 17) else 0
        
        features = pd.DataFrame([[hour, minute, day_of_week, distance_km, 
                                 call_type_num, rush_hour, distance_km_sq, 
                                 is_weekend, minute_of_day, distance_rush_interaction, 
                                 traffic_intensity]], columns=feature_columns)
        features = scaler.transform(features)
        
        # Predict
        model = models[selected_model]
        prediction = model.predict(features)[0] * 60
        logger.info(f"Prediction successful: Model={selected_model}, Prediction={prediction:.2f} minutes")
        
        return jsonify({
            'prediction': prediction,
            'user_inputs': {
                'Day of Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week],
                'Time': f"{hour:02d}:{minute:02d}",
                'Distance (km)': f"{distance_km:.2f}",
                'Call Type': call_type
            }
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
