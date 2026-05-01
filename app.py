"""
HeartWise AI — Model Inference Server
  POST /predict-tabular  → XGBClassifier, 13 features, binary (0=Low Risk, 1=High Risk)
  POST /predict-ecg      → EfficientNetB3 Keras model, binary (Normal / Abnormal)

Run:  python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, numpy as np, os, io, traceback
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

BASE = os.path.dirname(__file__)

TABULAR_MODEL_PATH = os.path.join(BASE, "heart_disease_model.pkl")
ECG_MODEL_PATH     = os.path.join(BASE, "ecg_model_best_75_accuracy")  # folder

# ── ECG config (confirmed from config.json) ────────────────────────────────
ECG_IMAGE_SIZE = (224, 224)
# ✏️  Update these to match your training labels (index 0 and index 1)
ECG_CLASSES    = ["Normal", "Abnormal"]

# ── Tabular feature order (confirmed from model.feature_names_in_) ─────────
# EXACTLY: age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ── Load tabular model ─────────────────────────────────────────────────────
# The .pkl is a plain XGBClassifier — NO scaler was saved with it.
# XGBoost handles raw feature values internally; no manual scaling needed.
print("Loading tabular model (XGBClassifier)...")
with open(TABULAR_MODEL_PATH, "rb") as f:
    tabular_model = pickle.load(f)
tabular_scaler = None   # confirmed: not present in this pkl
print(f"Tabular model loaded ✓  features={tabular_model.n_features_in_}  classes={tabular_model.classes_}")

# ── Load ECG Keras model ───────────────────────────────────────────────────
print("Loading ECG model (EfficientNetB3)...")
if not os.path.exists(ECG_MODEL_PATH):
    raise FileNotFoundError(
        f"\nECG model folder not found: {ECG_MODEL_PATH}\n"
        "Extract ecg_model_best_75_accuracy__keras.zip inside backend/ so you have:\n"
        "  backend/ecg_model_best_75_accuracy/\n"
        "      config.json | metadata.json | model.weights.h5"
    )
ecg_model = tf.keras.models.load_model(ECG_MODEL_PATH)
print(f"ECG model loaded ✓  output shape: {ecg_model.output_shape}")


def preprocess_ecg(pil_img: Image.Image) -> np.ndarray:
    """EfficientNetB3 needs preprocess_input (scales to [-1,1]), NOT /255."""
    img = pil_img.convert("RGB").resize(ECG_IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    return tf.keras.applications.efficientnet.preprocess_input(arr)


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — Tabular prediction  (XGBClassifier, no scaler)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/predict-tabular", methods=["POST"])
def predict_tabular():
    """
    Body:    { "features": [age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak, slope, ca, thal] }
    Returns: { "prediction": 0|1, "probability": 0.0-1.0, "label": "Low Risk"|"High Risk" }
    """
    try:
        body     = request.get_json(force=True)
        features = body.get("features")

        if not features or len(features) != len(FEATURE_ORDER):
            return jsonify({
                "error": f"Expected {len(FEATURE_ORDER)} features, got {len(features) if features else 0}"
            }), 400

        X = np.array(features, dtype=float).reshape(1, -1)
        # No scaler — pass raw values straight to XGBoost
        prediction  = int(tabular_model.predict(X)[0])
        probability = float(tabular_model.predict_proba(X)[0][1])  # P(High Risk)

        return jsonify({
            "prediction":  prediction,
            "probability": round(probability, 4),
            "label":       "High Risk" if prediction == 1 else "Low Risk",
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — ECG classification  (EfficientNetB3, 2 classes)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/predict-ecg", methods=["POST"])
def predict_ecg():
    """
    Multipart: { file: <image> }
    OR JSON:   { "image_url": "https://..." }

    Returns: { "label": "Normal", "confidence": 0.97,
               "all_classes": [{"label":"Normal","confidence":0.97}, ...] }
    """
    try:
        pil_img = None

        if "file" in request.files:
            pil_img = Image.open(request.files["file"].stream)
        elif request.is_json:
            import urllib.request as urlreq
            url = request.get_json(force=True).get("image_url")
            if not url:
                return jsonify({"error": "Provide 'file' or 'image_url'"}), 400
            with urlreq.urlopen(url) as resp:
                pil_img = Image.open(io.BytesIO(resp.read()))
        else:
            return jsonify({"error": "Provide multipart 'file' or JSON 'image_url'"}), 400

        arr   = preprocess_ecg(pil_img)
        preds = ecg_model.predict(np.expand_dims(arr, axis=0))[0]  # (2,)

        top   = int(np.argmax(preds))
        return jsonify({
            "label":       ECG_CLASSES[top] if top < len(ECG_CLASSES) else f"Class {top}",
            "confidence":  round(float(preds[top]), 4),
            "all_classes": [
                {"label":      ECG_CLASSES[i] if i < len(ECG_CLASSES) else f"Class {i}",
                 "confidence": round(float(preds[i]), 4)}
                for i in range(len(preds))
            ],
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# ── Health-check ───────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "tabular_model":  "XGBClassifier",
        "tabular_scaler": "none (not needed)",
        "ecg_classes":    ECG_CLASSES,
        "ecg_input_size": list(ECG_IMAGE_SIZE),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
