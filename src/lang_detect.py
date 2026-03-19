from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

# Ký tự có dấu tiếng Việt — dùng để detect chính xác hơn
VI_CHARS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")

def detect_language(text: str) -> str:
    text = text.strip()
    if not text:
        return "en"
    vi_ratio = sum(1 for c in text.lower() if c in VI_CHARS) / max(len(text), 1)
    if vi_ratio > 0.05:
        return "vi"
    if len(text) < 10:
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"

SYSTEM_PROMPTS = {
    "vi": "Bạn là trợ lý AI thông minh, thân thiện. Trả lời bằng tiếng Việt, ngắn gọn và chính xác.",
    "en": "You are a smart, friendly AI assistant. Answer concisely and accurately in English.",
    "zh-cn": "你是一个智能助手。请用简体中文简洁准确地回答。",
    "ja": "あなたは賢いAIアシスタントです。日本語で簡潔に答えてください。",
}

def get_system_prompt(lang: str) -> str:
    return SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])