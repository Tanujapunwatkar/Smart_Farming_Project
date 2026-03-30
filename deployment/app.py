import os
import sys
import uuid
import requests as req
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ❌ REMOVE tensorflow import (not needed anymore)
# import tensorflow as tf

# ─────────────────────────────────────────────
# ENV + PATH
# ─────────────────────────────────────────────

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

UI_PATH = os.path.join(BASE_DIR, "deployment", "farmer_ui")

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

from deployment.predict import predict_image
from chatbot.chatbot import FarmingChatbot
from ai_engine.solution_engine import get_full_solution

# ❌ REMOVE THIS ENTIRE BLOCK
# _model = None
# def get_model():
#     global _model
#     if _model is None:
#         _model = tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────

app = FastAPI(title="Smart Farming AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, FarmingChatbot] = {}

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

class GeneralChatRequest(BaseModel):
    message: str
    language: str = "English"

# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    index_file = os.path.join(UI_PATH, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "UI not found"}

# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────
# PREDICT (FIXED)
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    class_name: Optional[str] = Form(None),
    language: str = Form("English")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload image only")

    image_bytes = await file.read()

    try:
        # ✅ NO MODEL LOADING HERE
        result = predict_image(image_bytes, class_name)

        solution = get_full_solution(
            result["class_name"],
            result["confidence"],
            result["is_healthy"],
            language
        )

        result["ai_summary"] = solution["ai_summary"]

        session_id = str(uuid.uuid4())
        sessions[session_id] = FarmingChatbot(result)

        result["session_id"] = session_id
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────
# CHAT
# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    chatbot = sessions[request.session_id]
    response = chatbot.chat(request.message)

    return ChatResponse(
        session_id=request.session_id,
        response=response
    )

# ─────────────────────────────────────────────
# GENERAL CHAT
# ─────────────────────────────────────────────

@app.post("/general-chat")
async def general_chat(request: GeneralChatRequest):
    if not GEMINI_KEY:
        return {"response": "Hello Farmer! Ask me anything 🌾"}

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"

        body = {
            "contents": [{"parts": [{"text": request.message}]}]
        }

        res = req.post(url, json=body, timeout=15)
        data = res.json()

        return {
            "response": data["candidates"][0]["content"]["parts"][0]["text"]
        }

    except Exception:
        return {"response": "Sorry, try again."}

# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"history": sessions[session_id].get_history()}