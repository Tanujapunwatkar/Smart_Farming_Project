# deployment/predict.py

import os
import sys
import numpy as np
import cv2
import keras   # ✅ use keras (not tensorflow.keras)

from ai_engine.solution_engine import get_full_solution

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

IMG_SIZE   = (64, 64)
THRESHOLD  = 0.35

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus'
]

HEALTHY_CLASSES = [
    'Pepper__bell___healthy',
    'Potato___healthy',
    'Tomato_healthy'
]

_model = None

# ─────────────────────────────────────────────
# LOAD MODEL (HF DIRECT)
# ─────────────────────────────────────────────

def load_model():
    global _model

    if _model is None:
        print("⏳ Loading model from Hugging Face...")

        _model = keras.saving.load_model(
            "hf://Tanupunwatkar/Smart_Farming_model",
            compile=False   # ✅ avoids compatibility issues
        )

        print("✅ Model LOADED SUCCESSFULLY")

    return _model


# ─────────────────────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Cannot decode image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_image(image_bytes: bytes, class_name: str = None) -> dict:

    model = load_model()

    if model is None:
        return {"error": "Model not loaded"}

    # preprocess
    img_tensor = preprocess_image(image_bytes)

    # predict
    prob       = float(model.predict(img_tensor, verbose=0)[0][0])
    is_healthy = prob < THRESHOLD
    confidence = (1 - prob) * 100 if is_healthy else prob * 100

    # fallback class
    if class_name is None:
        class_name = 'Tomato_healthy' if is_healthy else 'Tomato_Late_blight'

    # AI solution
    solution = get_full_solution(class_name, confidence, is_healthy)

    return {
        'prediction'    : 'Healthy' if is_healthy else 'Diseased',
        'confidence'    : round(confidence, 2),
        'probability'   : round(prob, 4),
        'is_healthy'    : is_healthy,
        'class_name'    : class_name,
        'cause'         : solution['cause'],
        'severity'      : solution['severity'],
        'symptoms'      : solution['symptoms'],
        'treatment'     : solution['treatment'],
        'prevention'    : solution['prevention'],
        'recovery_days' : solution['recovery_days'],
        'ai_summary'    : solution['ai_summary']
    }