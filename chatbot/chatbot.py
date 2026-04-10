# chatbot/chatbot.py

import os
import requests
from chatbot.prompts import build_context_prompt

from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

API_KEY = os.getenv("GEMINI_API_KEY", "")
print(f"[Chatbot] GEMINI KEY: {'LOADED' if API_KEY else 'MISSING'}")

LANG_INSTRUCTIONS = {
    "English": "You MUST respond ONLY in English. Do not use any other language.",
    "Hindi"  : "आपको केवल हिंदी में जवाब देना है। कोई अन्य भाषा का उपयोग न करें। Always respond in Hindi only.",
    "Marathi": "तुम्ही फक्त मराठीत उत्तर द्यायला हवे. इतर कोणतीही भाषा वापरू नका. Always respond in Marathi only.",
}

GEMINI_MODEL = "gemini-2.5-flash"


class FarmingChatbot:

    def __init__(self, prediction_result: dict):
        self.prediction_result    = prediction_result
        self.conversation_history = []
        self.base_system_prompt   = build_context_prompt(prediction_result)
        self.current_lang         = "English"

    def chat(self, user_message: str) -> str:
        # extract language tag if present
        clean_message = user_message
        lang          = self.current_lang

        if "\n\n[Respond only in " in user_message:
            parts         = user_message.split("\n\n[Respond only in ")
            clean_message = parts[0].strip()
            lang_raw      = parts[1].replace("]", "").strip()
            if lang_raw in LANG_INSTRUCTIONS:
                lang              = lang_raw
                self.current_lang = lang

        self.conversation_history.append({
            "role"   : "user",
            "content": clean_message
        })

        if not API_KEY:
            response = self._fallback_response(clean_message, lang)
            self.conversation_history.append({
                "role": "assistant", "content": response
            })
            return response

        try:
            response = self._call_gemini(clean_message, lang)
            self.conversation_history.append({
                "role": "assistant", "content": response
            })
            return response

        except Exception as e:
            print(f"[Chatbot] Gemini error: {e}")
            response = self._fallback_response(clean_message, lang)
            self.conversation_history.append({
                "role": "assistant", "content": response
            })
            return response

    def _call_gemini(self, user_message: str, lang: str = "English") -> str:
        lang_instr = LANG_INSTRUCTIONS.get(lang, LANG_INSTRUCTIONS["English"])
        system     = f"{self.base_system_prompt}\n\n===LANGUAGE RULE===\n{lang_instr}"

        contents = []
        for msg in self.conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role" : role,
                "parts": [{"text": msg["content"]}]
            })

        url  = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"
        body = {
            "system_instruction": {
                "parts": [{"text": system}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature"    : 0.7,
                "maxOutputTokens": 1024
            }
        }

        res  = requests.post(url, json=body, timeout=20)
        res.raise_for_status()
        data = res.json()

        if "error" in data:
            raise Exception(data["error"].get("message", "Gemini API error"))

        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _fallback_response(self, user_message: str, lang: str = "English") -> str:
        msg        = user_message.lower()
        is_healthy = self.prediction_result.get("is_healthy", True)
        class_name = self.prediction_result.get("class_name", "")
        treatment  = self.prediction_result.get("treatment", [])
        prevention = self.prediction_result.get("prevention", [])
        severity   = self.prediction_result.get("severity", "Unknown")
        recovery   = self.prediction_result.get("recovery_days", "Unknown")

        # ── MARATHI ───────────────────────────────────────────
        if lang == "Marathi":
            if any(w in msg for w in ["नमस्कार","नमस्ते","hello","hi","hii","हॅलो"]):
                if is_healthy:
                    return (
                        "नमस्कार! मी तुमचा AI शेती सल्लागार आहे.\n\n"
                        "तुमचे झाड निरोगी आहे! मी मदत करू शकतो:\n"
                        "- पाणी आणि खताच्या टिप्स\n"
                        "- रोगापासून संरक्षण\n"
                        "- कोणताही शेतीचा प्रश्न\n\n"
                        "तुम्हाला काय जाणून घ्यायचे आहे?"
                    )
                name = class_name.replace("_", " ")
                return (
                    f"नमस्कार! मला {name} आढळला आहे.\n\n"
                    f"तीव्रता: {severity}\n"
                    f"बरे होण्याचा कालावधी: {recovery}\n\n"
                    f"उपचार किंवा प्रतिबंधाबद्दल विचारा!"
                )
            if any(w in msg for w in ["उपचार","treat","spray","औषध","medicine","fix","cure"]):
                if treatment:
                    steps = "\n".join([f"{i+1}. {t}" for i, t in enumerate(treatment)])
                    return f"उपचाराचे टप्पे:\n\n{steps}"
            if any(w in msg for w in ["टाळ","prevent","प्रतिबंध","रोख","भविष्य"]):
                if prevention:
                    tips = "\n".join([f"- {p}" for p in prevention])
                    return f"प्रतिबंधाचे उपाय:\n\n{tips}"
            if any(w in msg for w in ["पाणी","water","irrigat","सिंचन"]):
                return (
                    "पाण्याच्या टिप्स:\n\n"
                    "1. झाडाच्या मुळांशी पाणी द्या — पानांवर नको\n"
                    "2. सकाळी लवकर पाणी द्या\n"
                    "3. पाणी देण्यापूर्वी माती तपासा\n"
                    "4. दर 2-3 दिवसांनी पुरेसे असते"
                )
            if any(w in msg for w in ["बरे","recover","किती दिवस","वेळ","time"]):
                return f"बरे होण्याचा अपेक्षित कालावधी: {recovery}\n\nनियमितपणे उपचार करा."
            if any(w in msg for w in ["खाणे","फळ","खाऊ","eat","fruit","safe"]):
                if is_healthy:
                    return "तुमचे झाड निरोगी आहे — फळे खाणे सुरक्षित आहे. खाण्यापूर्वी नीट धुवा."
                return "रोगग्रस्त झाडाचे जास्त प्रभावित फळे खाणे टाळा. उपचारानंतर नवीन वाढीची प्रतीक्षा करा."
            if any(w in msg for w in ["खत","fertiliz","nutrient"]):
                return (
                    "खताच्या टिप्स:\n\n"
                    "1. संतुलित NPK खत वापरा\n"
                    "2. दर 2-3 आठवड्यांनी द्या\n"
                    "3. जास्त खत देणे टाळा\n"
                    "4. रोग उपचारादरम्यान नायट्रोजन कमी करा"
                )
            if any(w in msg for w in ["तीव्रता","serious","गंभीर","danger"]):
                return f"तीव्रता: {severity}\n\nत्वरित उपचार करा आणि वरील टप्पे पाळा."
            if is_healthy:
                return (
                    "तुमचे झाड निरोगी आहे! सामान्य काळजी:\n\n"
                    "1. नियमितपणे मुळांशी पाणी द्या\n"
                    "2. चांगला सूर्यप्रकाश मिळवा (6-8 तास)\n"
                    "3. दर आठवड्याला कीटकांची तपासणी करा\n"
                    "4. दर महिन्याला संतुलित खत वापरा\n"
                    "5. मृत पाने त्वरित काढा"
                )
            name = class_name.replace("_", " ")
            return (
                f"{name} साठी:\n\n"
                f"तीव्रता: {severity}\n"
                f"बरे होण्याचा कालावधी: {recovery}\n\n"
                f"वर दाखवलेले उपचाराचे टप्पे काळजीपूर्वक पाळा.\n"
                f"अधिक मदतीसाठी स्थानिक कृषी अधिकाऱ्याशी संपर्क करा."
            )

        # ── HINDI ─────────────────────────────────────────────
        elif lang == "Hindi":
            if any(w in msg for w in ["नमस्ते","नमस्कार","hello","hi","hii","हेलो"]):
                if is_healthy:
                    return (
                        "नमस्ते! मैं आपका AI कृषि सलाहकार हूं।\n\n"
                        "आपका पौधा स्वस्थ है! मैं मदद कर सकता हूं:\n"
                        "- पानी और खाद की सलाह\n"
                        "- पौधे को स्वस्थ कैसे रखें\n"
                        "- कोई भी कृषि सवाल\n\n"
                        "आप क्या जानना चाहते हैं?"
                    )
                name = class_name.replace("_", " ")
                return (
                    f"नमस्ते! मुझे {name} मिला है।\n\n"
                    f"गंभीरता: {severity}\n"
                    f"ठीक होने का समय: {recovery}\n\n"
                    f"उपचार या रोकथाम के बारे में पूछें!"
                )
            if any(w in msg for w in ["उपचार","treat","spray","दवाई","medicine","fix","cure"]):
                if treatment:
                    steps = "\n".join([f"{i+1}. {t}" for i, t in enumerate(treatment)])
                    return f"उपचार के कदम:\n\n{steps}"
            if any(w in msg for w in ["रोकथाम","prevent","अगली बार","भविष्य"]):
                if prevention:
                    tips = "\n".join([f"- {p}" for p in prevention])
                    return f"रोकथाम के उपाय:\n\n{tips}"
            if any(w in msg for w in ["पानी","water","irrigat","सिंचाई"]):
                return (
                    "पानी देने की सलाह:\n\n"
                    "1. पौधे की जड़ों में पानी दें — पत्तियों पर नहीं\n"
                    "2. सुबह जल्दी पानी दें\n"
                    "3. पानी देने से पहले मिट्टी की नमी जांचें\n"
                    "4. हर 2-3 दिन में पर्याप्त है"
                )
            if any(w in msg for w in ["ठीक","recover","कितने दिन","समय","time"]):
                return f"ठीक होने का अपेक्षित समय: {recovery}\n\nनियमित रूप से उपचार करें।"
            if any(w in msg for w in ["खाना","फल","खा","eat","fruit","safe"]):
                if is_healthy:
                    return "आपका पौधा स्वस्थ है — फल खाना सुरक्षित है। खाने से पहले अच्छी तरह धोएं।"
                return "ज्यादा प्रभावित फल खाने से बचें। उपचार के बाद नई स्वस्थ वृद्धि का इंतजार करें।"
            if any(w in msg for w in ["खाद","fertiliz","nutrient","उर्वरक"]):
                return (
                    "खाद की सलाह:\n\n"
                    "1. संतुलित NPK खाद का उपयोग करें\n"
                    "2. हर 2-3 हफ्ते में दें\n"
                    "3. अत्यधिक खाद से बचें\n"
                    "4. बीमारी के उपचार के दौरान नाइट्रोजन कम करें"
                )
            if any(w in msg for w in ["गंभीर","serious","खतरा","danger","बुरा"]):
                return f"गंभीरता: {severity}\n\nतुरंत उपचार करें और ऊपर के कदम पालन करें।"
            if is_healthy:
                return (
                    "आपका पौधा स्वस्थ है! सामान्य देखभाल:\n\n"
                    "1. जड़ों में नियमित पानी दें\n"
                    "2. अच्छी धूप सुनिश्चित करें (6-8 घंटे)\n"
                    "3. हर हफ्ते कीटों की जांच करें\n"
                    "4. हर महीने संतुलित खाद दें\n"
                    "5. मृत पत्तियां तुरंत हटाएं"
                )
            name = class_name.replace("_", " ")
            return (
                f"{name} के लिए:\n\n"
                f"गंभीरता: {severity}\n"
                f"ठीक होने का समय: {recovery}\n\n"
                f"ऊपर दिखाए गए उपचार के कदम ध्यान से पालन करें।\n"
                f"अधिक सहायता के लिए स्थानीय कृषि अधिकारी से संपर्क करें।"
            )

        # ── ENGLISH (default) ─────────────────────────────────
        else:
            if any(w in msg for w in ["hi","hello","hey","hii","namaste"]):
                if is_healthy:
                    return (
                        "Hello! I am your AI farming advisor.\n\n"
                        "Your plant is healthy! I can help with:\n"
                        "- Watering and fertilization tips\n"
                        "- How to keep your plant disease-free\n"
                        "- Any farming questions\n\n"
                        "What would you like to know?"
                    )
                name = class_name.replace("_", " ")
                return (
                    f"Hello! I detected {name}.\n\n"
                    f"Severity: {severity}\n"
                    f"Recovery: {recovery}\n\n"
                    f"Ask me about treatment or prevention!"
                )
            if any(w in msg for w in ["treat","fix","cure","apply","spray","medicine"]):
                if treatment:
                    steps = "\n".join([f"{i+1}. {t}" for i, t in enumerate(treatment)])
                    return f"Treatment steps:\n\n{steps}"
            if any(w in msg for w in ["prevent","avoid","future","again","next time"]):
                if prevention:
                    tips = "\n".join([f"- {p}" for p in prevention])
                    return f"Prevention tips:\n\n{tips}"
            if any(w in msg for w in ["water","irrigat","when to water"]):
                return (
                    "Watering tips:\n\n"
                    "1. Water at the base — not on leaves\n"
                    "2. Water early morning\n"
                    "3. Check soil moisture before watering\n"
                    "4. Every 2-3 days is usually enough"
                )
            if any(w in msg for w in ["recover","heal","long","days","time","when"]):
                return f"Expected recovery: {recovery}\n\nBe consistent with treatment."
            if any(w in msg for w in ["eat","fruit","safe","consume","food"]):
                if is_healthy:
                    return "Your plant is healthy — fruit is safe to eat. Always wash before eating."
                return "Avoid eating heavily infected fruit. Wait for healthy regrowth after treatment."
            if any(w in msg for w in ["fertiliz","nutrient","feed","compost"]):
                return (
                    "Fertilization tips:\n\n"
                    "1. Use balanced NPK fertilizer\n"
                    "2. Apply every 2-3 weeks\n"
                    "3. Avoid over-fertilizing\n"
                    "4. Reduce nitrogen during disease treatment\n"
                    "5. Add compost to improve soil health"
                )
            if any(w in msg for w in ["severe","serious","danger","bad","worse"]):
                return f"Severity: {severity}\n\nAct quickly and follow treatment steps carefully."
            if is_healthy:
                return (
                    "Your plant is healthy! General care:\n\n"
                    "1. Water regularly at the base\n"
                    "2. Ensure good sunlight (6-8 hours)\n"
                    "3. Check for pests weekly\n"
                    "4. Use balanced fertilizer monthly\n"
                    "5. Remove dead leaves promptly"
                )
            name = class_name.replace("_", " ")
            return (
                f"For {name}:\n\n"
                f"Severity: {severity}\n"
                f"Recovery: {recovery}\n\n"
                f"Follow the treatment steps shown carefully.\n"
                f"Consult your local agricultural officer if symptoms worsen."
            )

    def reset(self):
        self.conversation_history = []

    def get_history(self) -> list:
        return self.conversation_history