import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import soundfile as sf
import io

# Import feature extraction function
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

# --- Load models (models should be present after Docker build) ---
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_scaler = joblib.load(RF_SCALER_PATH)
    print("✅ โหลดโมเดล RandomForest และ Scaler เรียบร้อยแล้ว")
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดลหรือ Scaler ได้: {e}")
    raise e

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "ไม่พบไฟล์เสียงในคำขอ"}), 400
    
    audio_file = request.files['file']
    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Read audio from request
        audio_data, sr = sf.read(io.BytesIO(audio_file.read()))

        # Save as WAV
        sf.write(filepath, audio_data, sr, format='WAV')

        # Extract features
        features = extract_features_rf(filepath)
        if features is None:
            return jsonify({"error": "ไม่สามารถสกัดฟีเจอร์จากไฟล์เสียงได้"}), 500

        # Scale features
        features_scaled = rf_scaler.transform(features.reshape(1, -1))

        # Predict
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
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
