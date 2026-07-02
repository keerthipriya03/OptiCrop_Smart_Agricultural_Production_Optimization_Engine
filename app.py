import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Safely load the saved model and scaler

MODEL_PATH = 'models/crop_model.joblib'
SCALER_PATH = 'models/scaler.joblib'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Crop model and scaler loaded successfully!")
else:
    model = None
    scaler = None
    print("WARNING: Model or Scaler file not found! Run train.py first.")

@app.route('/')
def home():
    """Renders the single page application interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on the input features."""
    if model is None or scaler is None:
        return jsonify({'error': 'Machine learning model files not loaded.'}), 500


    try:
        data = request.get_json()

        # Required fields
        required_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing parameter: {key}'}), 400

    # Build feature array
        raw_features = np.array([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])

        # Simple parameter boundaries validation
        n, p, k, temp, hum, ph, rain = raw_features[0]
        if not (0 <= n <= 250): return jsonify({'error': 'N content must be between 0 and 250 mg/kg.'}), 400
        if not (0 <= p <= 250): return jsonify({'error': 'P content must be between 0 and 250 mg/kg.'}), 400
        if not (0 <= k <= 300): return jsonify({'error': 'K content must be between 0 and 300 mg/kg.'}), 400
        if not (-10 <= temp <= 60): return jsonify({'error': 'Temperature must be between -10°C and 60°C.'}), 400
        if not (0 <= hum <= 100): return jsonify({'error': 'Humidity must be between 0% and 100%.'}), 400
        if not (0 <= ph <= 14): return jsonify({'error': 'Soil pH must be between 0.0 and 14.0.'}), 400
        if not (0 <= rain <= 1000): return jsonify({'error': 'Rainfall must be between 0mm and 1000mm.'}), 400


        # Scale input
        scaled_features = scaler.transform(raw_features)

        # Predict
        prediction = model.predict(scaled_features)[0]

        return jsonify({
            'success': True,
            'crop': str(prediction).capitalize()
        })

    except ValueError:
        return jsonify({'error': 'Inputs must be valid numbers'}), 400
    except Exception as e:
        return jsonify({'error': f'Server Error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
