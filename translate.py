# translation.py
import requests

TRANSLATE_MODEL = "qwen3.5:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def translate(text: str, target_lang: str = "ja") -> str:
    """
    target_lang: 'ja' | 'zh-tw' | 'en'
    """
    lang_map = {"ja": "日本語", "zh-tw": "繁體中文", "en": "English"}
    prompt = (
        f"Translate the following text to {lang_map[target_lang]}. "
        f"Output ONLY the translation, no explanation, no notes.\n\n{text}"
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": TRANSLATE_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "0",   # 用完立刻釋放 VRAM
        "options": {"temperature": 0.1, "num_predict": 300, "num_ctx": 2048}
    }, timeout=30)
    return resp.json().get("response", "").strip()