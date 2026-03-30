# ai_engine/solution_engine.py

import os
import json
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
load_dotenv() 
KB_PATH = os.path.join(BASE_DIR, "ai_engine", "knowledge_base.json")
API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

print(f"[Solution Engine] GEMINI KEY: {'LOADED' if API_KEY else 'MISSING'}")

LANG_INSTRUCTIONS = {
    "English": "Respond only in English.",
    "Hindi"  : "केवल हिंदी में जवाब दें। Always respond in Hindi only.",
    "Marathi": "फक्त मराठीत उत्तर द्या. Always respond in Marathi only.",
}

# ─────────────────────────────────────────────
#  LOAD KNOWLEDGE BASE
# ─────────────────────────────────────────────

def load_knowledge_base() -> dict:
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# ─────────────────────────────────────────────
#  MAP CLASS TO KB KEY
# ─────────────────────────────────────────────

def map_class_to_kb_key(class_name: str) -> str:
    mapping = {
        'Tomato_Bacterial_spot'                       : 'Tomato_Bacterial_spot',
        'Tomato_Early_blight'                         : 'Tomato_Early_blight',
        'Tomato_Late_blight'                          : 'Tomato_Late_blight',
        'Tomato_Leaf_Mold'                            : 'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot'                   : 'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite' : 'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Target_Spot'                         : 'Tomato__Target_Spot',
        'Tomato__Tomato_mosaic_virus'                 : 'Tomato__Tomato_mosaic_virus',
        'Tomato__Tomato_YellowLeaf__Curl_Virus'       : 'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Potato___Early_blight'                       : 'Potato___Early_blight',
        'Potato___Late_blight'                        : 'Potato___Late_blight',
        'Pepper__bell___Bacterial_spot'               : 'Pepper__bell___Bacterial_spot',
        'Pepper__bell___healthy'                      : 'healthy',
        'Potato___healthy'                            : 'healthy',
        'Tomato_healthy'                              : 'healthy',
    }
    return mapping.get(class_name, 'healthy')

# ─────────────────────────────────────────────
#  GET KB SOLUTION
# ─────────────────────────────────────────────

def get_kb_solution(class_name: str) -> dict:
    kb     = load_knowledge_base()
    kb_key = map_class_to_kb_key(class_name)
    return kb.get(kb_key, kb['healthy'])

# ─────────────────────────────────────────────
#  CALL GEMINI VIA HTTP
# ─────────────────────────────────────────────

def call_gemini(prompt: str, language: str = "English") -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

    lang_instr = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["English"])

    system = f"""You are an expert agricultural advisor helping farmers in India.
Give practical simple advice that farmers can easily understand.
{lang_instr}"""

    body = {
        "system_instruction": {
            "parts": [{"text": system}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature"    : 0.7,
            "maxOutputTokens": 1024
        }
    }

    res  = requests.post(url, json=body, timeout=20)
    res.raise_for_status()
    data = res.json()

    if "error" in data:
        raise Exception(data["error"].get("message", "Gemini error"))

    return data["candidates"][0]["content"]["parts"][0]["text"]

# ─────────────────────────────────────────────
#  GET AI SOLUTION
# ─────────────────────────────────────────────

def get_ai_solution(class_name: str,
                    confidence: float,
                    is_healthy: bool,
                    language  : str = "English") -> str:
    if not API_KEY:
        return get_kb_solution_text(class_name, is_healthy)

    lang_instr = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS["English"])

    try:
        kb_data = get_kb_solution(class_name)

        if is_healthy:
            prompt = f"""The farmer's plant is HEALTHY (confidence: {confidence:.1f}%).

Give a warm encouraging response. Include:
1. Confirmation the plant is healthy
2. 3-4 tips to keep it healthy
3. Signs of disease to watch out for

Keep it short, practical and friendly.
{lang_instr}"""

        else:
            prompt = f"""The farmer's plant has been diagnosed with: {class_name.replace('_', ' ')}
Confidence: {confidence:.1f}%

Disease information:
- Cause: {kb_data['cause']}
- Severity: {kb_data['severity']}
- Symptoms: {kb_data['symptoms']}
- Expected recovery: {kb_data['recovery_days']}

Give a clear actionable response. Include:
1. Brief explanation of the disease
2. How serious it is
3. Step-by-step treatment (numbered list)
4. Prevention tips
5. Expected recovery time

Keep it practical and friendly. No technical jargon.
{lang_instr}"""

        return call_gemini(prompt, language)

    except Exception as e:
        print(f"[Solution Engine] Gemini error: {e}")
        return get_kb_solution_text(class_name, is_healthy)

# ─────────────────────────────────────────────
#  FORMAT KB SOLUTION AS TEXT (offline fallback)
# ─────────────────────────────────────────────

def get_kb_solution_text(class_name: str, is_healthy: bool) -> str:
    data = get_kb_solution(class_name)

    if is_healthy:
        tips = "\n".join([f"  - {t}" for t in data['prevention']])
        return (
            f"Your plant looks healthy! Great job.\n\n"
            f"Tips to keep it healthy:\n{tips}\n\n"
            f"Keep monitoring regularly."
        )

    name       = class_name.replace('_', ' ').replace('  ', ' ')
    treatment  = "\n".join([f"  {i+1}. {t}" for i, t in enumerate(data['treatment'])])
    prevention = "\n".join([f"  - {p}" for p in data['prevention']])

    return (
        f"Disease Detected: {name}\n\n"
        f"Cause: {data['cause']}\n"
        f"Severity: {data['severity']}\n"
        f"Symptoms: {data['symptoms']}\n\n"
        f"Treatment Steps:\n{treatment}\n\n"
        f"Prevention:\n{prevention}\n\n"
        f"Expected Recovery: {data['recovery_days']}"
    )

# ─────────────────────────────────────────────
#  FULL SOLUTION PACKAGE
# ─────────────────────────────────────────────

def get_full_solution(class_name: str,
                      confidence: float,
                      is_healthy: bool,
                      language  : str = "English") -> dict:
    kb_data = get_kb_solution(class_name)
    ai_text = get_ai_solution(class_name, confidence, is_healthy, language)

    return {
        'class_name'   : class_name,
        'is_healthy'   : is_healthy,
        'confidence'   : confidence,
        'cause'        : kb_data['cause'],
        'severity'     : kb_data['severity'],
        'symptoms'     : kb_data['symptoms'],
        'treatment'    : kb_data['treatment'],
        'prevention'   : kb_data['prevention'],
        'recovery_days': kb_data['recovery_days'],
        'ai_summary'   : ai_text
    }