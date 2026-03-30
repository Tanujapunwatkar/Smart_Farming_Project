# deployment/predict.py

import numpy as np
import cv2
import keras
from huggingface_hub import hf_hub_download

from ai_engine.solution_engine import get_full_solution

IMG_SIZE = (64, 64)
THRESHOLD = 0.35

_model = None


# ✅ LOAD MODEL (FINAL FIX)
def load_model():
    global _model

    if _model is None:
        print("⏳ Downloading model from Hugging Face...")

        try:
            model_path = hf_hub_download(
                repo_id="Tanupunwatkar/Smart_Farming_model",
                filename="best_model.keras"
            )

            print("📁 Model downloaded at:", model_path)

            _model = keras.models.load_model(
                model_path,
                compile=False
            )

        except Exception as e:
            print("❌ Error loading model:", e)
            raise RuntimeError("Model loading failed")

        print("✅ Model LOADED SUCCESSFULLY")

    return _model


# ✅ PREPROCESS
def preprocess_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ✅ PREDICT
def predict_image(image_bytes: bytes, class_name=None):

    model = load_model()
    img = preprocess_image(image_bytes)

    prob = float(model.predict(img, verbose=0)[0][0])

    is_healthy = prob < THRESHOLD
    confidence = (1 - prob) * 100 if is_healthy else prob * 100

    if class_name is None:
        class_name = "Tomato_healthy" if is_healthy else "Tomato_Late_blight"

    solution = get_full_solution(class_name, confidence, is_healthy)

    return {
        "prediction": "Healthy" if is_healthy else "Diseased",
        "confidence": round(confidence, 2),
        "class_name": class_name,
        "is_healthy": is_healthy,
        "ai_summary": solution["ai_summary"]
    }