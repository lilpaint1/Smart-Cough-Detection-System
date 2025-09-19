import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import soundfile as sf
import io

from rf_extract import extract_features_rf

# --- Constants ---
LABELS = ['covid', 'healthy', 'symptomatic']
RF_MODEL_PATH = "cough_rf_model.pkl"
RF_SCALER_PATH = "scaler_rf.pkl"
UPLOAD_FOLDER = "uploads"

# --- Create Flask app ---
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load models ---
# Gunicorn will run this part when the container starts
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_scaler = joblib.load(RF_SCALER_PATH)
    print("✅ Load models and scalers successfully")
except Exception as e:
    print(f"❌ Failed to load models or scalers: {e}")
    # Raise an error to make the container fail and show in logs
    raise e

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file found in the request"}), 400

    audio_file = request.files['file']
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        audio_data, sr = sf.read(io.BytesIO(audio_file.read()))
        sf.write(filepath, audio_data, sr, format='WAV')

        features = extract_features_rf(filepath)
        if features is None:
            return jsonify({"error": "Failed to extract features from audio"}), 500

        features_scaled = rf_scaler.transform(features.reshape(1, -1))
        proba = rf_model.predict_proba(features_scaled)[0]
        pred_idx = int(np.argmax(proba))
        predicted_label = LABELS[pred_idx]

        probabilities = [
            {"label": lbl, "score": float(prob)}
            for lbl, prob in zip(LABELS, proba)
        ]

        return jsonify({
            "classification": predicted_label,
            "probabilities": probabilities
        }), 200

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
