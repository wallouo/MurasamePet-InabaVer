import os
import requests
import json

print("🔍 正在執行 Coder (v1/responses) API 結構診斷...")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ 找不到 OPENAI_API_KEY 環境變數！")
    exit()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
url = "https://api.openai.com/v1/responses"

def run_test(test_name, payload):
    print(f"\n▶️ [{test_name}] 正在發送請求...")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        print(f"📥 狀態碼: {resp.status_code}")
        try:
            # 嘗試格式化印出 JSON，方便閱讀錯誤訊息
            parsed_resp = resp.json()
            print(f"📄 回傳內容:\n{json.dumps(parsed_resp, indent=2, ensure_ascii=False)}")
        except:
            print(f"📄 回傳內容: {resp.text}")
    except Exception as e:
        print(f"❌ [請求失敗]: {type(e).__name__} - {e}")

# ==========================================
# 測試 A：完全不加 format 參數 (最安全的 Baseline)
# ==========================================
payload_a = {
    "model": "gpt-5-codex",
    "input": [
        {"role": "user", "content": "Reply 'OK' in JSON format like {\"code\": \"OK\"}."}
    ]
}
run_test("測試 A - 移除所有 format 參數", payload_a)

# ==========================================
# 測試 B：使用 text.format 嵌套結構
# ==========================================
payload_b = {
    "model": "gpt-5-codex",
    "text": {
        "format": "json_object"
    },
    "input": [
        {"role": "user", "content": "Reply 'OK' in JSON format like {\"code\": \"OK\"}."}
    ]
}
run_test("測試 B - 使用 text: {format: json_object}", payload_b)

print("\n--------------------------------------------------")
print("✅ 測試完畢，請將終端機印出的結果貼給我看！")