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

import os
import re
from groq import Groq

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Medical knowledge base ────────────────────────────────────────────────────
# Paste your PDF text here. The more content, the better the chatbot.
# You can also load from a .txt file: open("knowledge.txt").read()
MEDICAL_CONTEXT = """
Heart Disease Overview:
Heart disease refers to several types of heart conditions. The most common type is coronary artery disease (CAD), 
which can cause heart attack. Other types include heart failure, arrhythmia, and heart valve disease.

Risk Factors:
- High blood pressure (hypertension): systolic > 130 mmHg or diastolic > 80 mmHg
- High cholesterol: total cholesterol > 200 mg/dL is borderline high, > 240 mg/dL is high
- Diabetes: fasting blood sugar > 126 mg/dL
- Smoking: increases risk by 2-4x
- Obesity: BMI > 30
- Physical inactivity
- Family history of heart disease
- Age: men > 45, women > 55

Chest Pain Types (cp):
- Type 1 (Typical angina): chest pain caused by reduced blood flow to the heart
- Type 2 (Atypical angina): chest discomfort not fitting typical pattern
- Type 3 (Non-anginal pain): chest pain not related to heart
- Type 4 (Asymptomatic): no chest pain

ECG Findings:
- Normal: regular rhythm, normal ST segment
- Abnormal: ST depression, T-wave changes, left ventricular hypertrophy
- ST depression (oldpeak): > 2mm is clinically significant
- Thalassemia types: normal (3), fixed defect (6), reversible defect (7)

Medications commonly prescribed:
- Statins (atorvastatin, rosuvastatin): for high cholesterol
- ACE inhibitors (lisinopril, ramipril): for high blood pressure and heart failure
- Beta-blockers (metoprolol, carvedilol): for angina, high BP, heart failure
- Aspirin: antiplatelet, reduces clot risk in heart disease
- Nitrates (nitroglycerin): for angina relief
- Diuretics (furosemide): for heart failure fluid management
- Anticoagulants (warfarin, rivaroxaban): for atrial fibrillation

Lifestyle Recommendations:
- Diet: Mediterranean diet, reduce sodium < 2300mg/day, avoid saturated fats
- Exercise: 150 minutes moderate aerobic activity per week
- Weight: maintain BMI 18.5-24.9
- Smoking cessation: most important modifiable risk factor
- Alcohol: limit to 1 drink/day women, 2 drinks/day men
- Stress management: meditation, yoga, adequate sleep 7-9 hours

When to seek emergency care:
- Chest pain lasting > 15 minutes
- Pain spreading to arm, jaw, back
- Shortness of breath with chest pain
- Sudden dizziness or loss of consciousness
- Call emergency services immediately

Thalach (Maximum Heart Rate):
- Normal max heart rate = 220 - age
- Low thalach relative to age suggests chronotropic incompetence
- Exercise-induced angina (exang=1) with low thalach is high risk indicator

Blood Pressure Interpretation:
- Normal: < 120/80 mmHg
- Elevated: 120-129 / < 80 mmHg  
- Stage 1 hypertension: 130-139 / 80-89 mmHg
- Stage 2 hypertension: >= 140 / >= 90 mmHg
- Hypertensive crisis: > 180 / > 120 mmHg

Cholesterol Interpretation:
- Desirable: < 200 mg/dL
- Borderline high: 200-239 mg/dL
- High: >= 240 mg/dL
- LDL goal for high risk patients: < 70 mg/dL
"""

# ── Simple keyword-based retrieval ────────────────────────────────────────────
def retrieve_relevant_context(question: str, top_chars: int = 1500) -> str:
    """Return the most relevant sections from MEDICAL_CONTEXT based on keywords."""
    question_lower = question.lower()
    keywords = re.findall(r'\b\w{4,}\b', question_lower)  # words >= 4 chars
    
    paragraphs = [p.strip() for p in MEDICAL_CONTEXT.split('\n\n') if p.strip()]
    scored = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for kw in keywords if kw in para_lower)
        scored.append((score, para))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Always include top scored paragraphs up to top_chars
    result = []
    total = 0
    for score, para in scored:
        if total + len(para) > top_chars:
            break
        result.append(para)
        total += len(para)
    
    return '\n\n'.join(result) if result else MEDICAL_CONTEXT[:top_chars]


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — AI Chatbot  (Groq LLM + RAG)
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST"])
def chat():
    """
    Body: {
        "message": "user question",
        "history": [{"role": "user"|"assistant", "content": "..."}],  // optional
        "patient_context": {  // optional - pass prediction results
            "risk_label": "High Risk",
            "probability": 0.87,
            "ecg_result": "Abnormal"
        }
    }
    Returns: { "reply": "..." }
    """
    try:
        body = request.get_json(force=True)
        user_message = body.get("message", "").strip()
        history = body.get("history", [])
        patient_context = body.get("patient_context", {})

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Retrieve relevant medical knowledge
        relevant_knowledge = retrieve_relevant_context(user_message)

        # Build patient context string
        patient_info = ""
        if patient_context:
            patient_info = f"""
Current Patient Assessment:
- Heart Disease Risk: {patient_context.get('risk_label', 'Not assessed')} 
  (probability: {patient_context.get('probability', 'N/A')})
- ECG Result: {patient_context.get('ecg_result', 'Not assessed')}
"""

        system_prompt = f"""You are HeartWise AI, a knowledgeable cardiac health assistant integrated into a 
heart disease prediction platform. You help patients understand their heart health results and provide 
evidence-based medical information.

IMPORTANT RULES:
1. Always recommend consulting a real doctor for diagnosis and treatment decisions
2. You CAN suggest common medications used for heart conditions, but always say "your doctor may prescribe" 
3. Be empathetic, clear, and avoid overly technical jargon
4. If asked about emergency symptoms, always say call emergency services immediately
5. Base answers on the medical knowledge provided below
6. Keep responses concise (3-5 sentences max) unless a detailed explanation is needed

{patient_info}

MEDICAL KNOWLEDGE BASE (use this to answer questions):
{relevant_knowledge}

If asked something outside heart health, politely redirect to cardiac topics."""

        # Build messages for Groq
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 6 exchanges to stay within context)
        for h in history[-12:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        
        messages.append({"role": "user", "content": user_message})

        # Call Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Free, very high limits
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        return jsonify({"reply": reply})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Chat service unavailable. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
