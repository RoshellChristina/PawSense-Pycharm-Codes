from flask import Flask, request, jsonify
from flask_cors import CORS
import os, io, json, logging
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dog_emotion_classifier.h5")
model = load_model(MODEL_PATH)

# Load class indices saved during training
CI_PATH = os.path.join(os.path.dirname(__file__), "class_indices.json")
with open(CI_PATH, "r") as f:
    saved_ci = json.load(f)

class_indices = {v: k for k, v in saved_ci.items()}


# --------------------- Preprocessing ---------------------
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((600, 600))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


# --------------------- Prediction Endpoint ---------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    # Open image and ensure RGB
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    input_tensor = preprocess(img)

    # Make prediction
    preds = model.predict(input_tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = class_indices.get(idx, "unknown")
    confidence = float(preds[idx])

    return jsonify({
        "emotion": label,
        "confidence": confidence
    })

MODEL_DIR = os.path.join(os.path.dirname(__file__), "disease_prediction")

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SYMPTOMS_JSON = os.path.join(MODEL_DIR, "symptoms.json")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")

logger.info("Loading disease detection artifacts...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
with open(SYMPTOMS_JSON, "r", encoding="utf-8") as f:
    symptom_cols = json.load(f)
disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

def normalize_symptom_list(symptoms_str):
    if symptoms_str is None:
        return []
    return [s.strip() for s in str(symptoms_str).split(";") if s.strip()]

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    categorical_cols = ["Breed", "Size", "Sex", "AgeCategory", "VaccinationStatus"]
    numeric_cols = ["DiseaseDurationDays"]

    row = {}
    for c in categorical_cols:
        row[c] = payload.get(c, "") or ""

    for n in numeric_cols:
        val = payload.get(n, None)
        row[n] = val if val is not None else 0

    provided_symptoms_raw = normalize_symptom_list(payload.get("Symptoms", ""))
    provided_symptoms = [s.lower() for s in provided_symptoms_raw]

    # Map each symptom column to 1/0 using normalized comparison
    for sym in symptom_cols:
        norm = sym.strip().lower()
        row[sym] = 1 if norm in provided_symptoms else 0

    cols_in_order = categorical_cols + numeric_cols + symptom_cols
    df = pd.DataFrame([row], columns=cols_in_order)

    for n in numeric_cols:
        df[n] = pd.to_numeric(df[n], errors="coerce").fillna(0).astype(float)

    try:
        X_trans = preprocessor.transform(df)
    except Exception as e:
        logger.exception("Preprocessor transform failed")
        return jsonify({"error": "Preprocessor transform failed", "details": str(e)}), 500

    if hasattr(X_trans, "toarray"):
        X_np = X_trans.toarray()
    else:
        X_np = np.asarray(X_trans)

    X_np = X_np.astype(np.float32)

    try:
        probs = disease_model.predict(X_np, verbose=0)
    except Exception as e:
        logger.exception("Model prediction failed")
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    top_idx = int(np.argmax(probs[0]))
    top_prob = float(probs[0][top_idx])
    try:
        top_label = str(le.inverse_transform([top_idx])[0])
    except Exception:
        top_label = str(top_idx)

    return jsonify({
        "prediction": top_label,
        "probability": top_prob
    })


# Load pipeline (make sure this file exists next to this script)
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "pipeline.pkl")
logger.info("Loading pipeline from %s", PIPELINE_PATH)
pipeline = joblib.load(PIPELINE_PATH)


@app.route("/predict_score", methods=["POST"])
def predict_score():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid/missing JSON body"}), 400

    try:
        df = pd.DataFrame([data])
        pred = pipeline.predict(df)
        return jsonify({"total_score": float(pred[0])})
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": "prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    # Use 0.0.0.0 in production or Docker; localhost is fine for dev
    app.run(host="0.0.0.0", port=5000, debug=True)

