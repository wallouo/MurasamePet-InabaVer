"""
健康檢查腳本。

檢查 Ollama 和 VOICEVOX 服務是否可用。若 VOICEVOX 不可達，會提示將 TTS 落回 mock 模式。
同時列印出可用的 API 端點。
"""

import os
import sys
import requests
from urllib.parse import urlparse


def check_endpoint(url: str, path: str) -> bool:
    try:
        resp = requests.get(f"{url}{path}", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def main() -> int:
    ollama = os.getenv('OLLAMA_ENDPOINT', 'http://127.0.0.1:11434')
    voicevox = os.getenv('VOICEVOX_ENDPOINT', 'http://127.0.0.1:50021')
    ok_ollama = check_endpoint(ollama, '/api/tags')
    ok_voicevox = check_endpoint(voicevox, '/version')
    print(f"[HealthCheck] Ollama ({ollama}/api/tags):", 'OK' if ok_ollama else 'FAIL')
    if ok_voicevox:
        print(f"[HealthCheck] VOICEVOX ({voicevox}/version): OK")
    else:
        print(f"[HealthCheck] VOICEVOX ({voicevox}/version): WARN -> will use mock TTS")
    print("Available API endpoints:")
    api_port = os.getenv('API_PORT', '5000')
    base = f"http://127.0.0.1:{api_port}"
    for path in ["/qwen3", "/reply_bi", "/tts", "/say", "/pat"]:
        print(f"  {base}{path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())