import os
import requests

print("🔍 正在執行 Gemini API 連線測試...")

# 1. 檢查 API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ 找不到 GEMINI_API_KEY 環境變數！")
    exit()
else:
    print(f"✅ 成功讀取 API Key (開頭為: {api_key[:5]}...)")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash:generateContent?key={api_key}"
payload = {"contents": [{"role": "user", "parts": [{"text": "Hello, reply 'OK' if you see this."}]}]}

print("🌐 正在發送請求至 Google 伺服器 (等待時間 10 秒)...")

try:
    # 這裡我們不包裝任何重試機制，直接硬上
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
    print(f"📥 收到 HTTP 狀態碼: {resp.status_code}")
    print(f"📄 回傳內容: {resp.text}")
    
except requests.exceptions.Timeout:
    print("❌ [連線超時] 請求超過 10 秒沒有回應。這通常代表你的網路無法直連 Google。")
except requests.exceptions.ConnectionError as e:
    print("❌ [連線拒絕/阻擋] 無法建立連線。")
    print(f"詳細錯誤: {e}")
except Exception as e:
    print(f"❌ [其他錯誤]: {type(e).__name__} - {e}")