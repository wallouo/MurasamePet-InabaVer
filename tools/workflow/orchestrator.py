import os
import re
import random
import json
import requests
import ast
import threading
from typing import Optional, Dict, Any

# --- 路徑自動定位 ---
# 1. 取得目前腳本所在的目錄 (tools/workflow)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 推導出專案根目錄 (往上推兩層: workflow -> tools -> 根目錄)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# 3. 指定工具設定檔的絕對路徑 (放在 workflow 資料夾內)
LESSONS_FILE = os.path.join(SCRIPT_DIR, "lessons_learned.json")
CONTEXT_FILE = os.path.join(SCRIPT_DIR, "context.md")
# --- 環境變數 ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not GEMINI_KEY or not OPENAI_KEY:
    raise SystemExit("❌ 找不到 API Key。請確保已設定環境變數 GEMINI_API_KEY 與 OPENAI_API_KEY。")

HEADERS_OPENAI = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}

# ------------- 基礎工具與錯題本 (Persistent Memory) -------------
def retry_with_backoff(call_fn, tries=3, base=1.0, max_delay=30.0):
    import time
    resp = None
    for attempt in range(1, tries + 1):
        resp = call_fn()
        if resp is None:
            return None
        if resp.status_code in (200, 201):
            return resp
        if resp.status_code in (429, 502, 503, 504):
            wait = min(max_delay, base * (2 ** (attempt - 1)) * (1 + random.random() * 0.1))
            print(f"⏳ [API 忙碌] 狀態碼 {resp.status_code}，等待 {wait:.1f} 秒後重試 (第 {attempt}/{tries} 次)...")
            time.sleep(wait)
        else:
            return resp

def load_context() -> str:
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return "No context provided."

def load_lessons() -> str:
    """🌟 載入歷史教訓"""
    if os.path.exists(LESSONS_FILE):
        try:
            with open(LESSONS_FILE, "r", encoding="utf-8") as f:
                lessons = json.load(f)
                if lessons:
                    return "\n".join([f"- {l}" for l in lessons])
        except Exception:
            pass
    return "目前無歷史教訓。"

def save_lesson(lesson: str):
    """🌟 儲存新教訓到錯題本"""
    lessons = []
    if os.path.exists(LESSONS_FILE):
        try:
            with open(LESSONS_FILE, "r", encoding="utf-8") as f:
                lessons = json.load(f)
        except Exception:
            pass
    lessons.append(lesson)
    with open(LESSONS_FILE, "w", encoding="utf-8") as f:
        json.dump(lessons, f, indent=2, ensure_ascii=False)

def safe_ast_check(code: str) -> Optional[str]:
    try:
        ast.parse(code)
        return None
    except Exception as e:
        return str(e)

def dangerous_pattern_scan(code: str) -> Optional[str]:
    DANGEROUS = [r"\beval\s*\(", r"\bexec\s*\(", r"os\.system", r"subprocess", r"rm\s+-rf"]
    found = [p for p in DANGEROUS if re.search(p, code)]
    return ", ".join(found) if found else None

# ------------- Agents -------------

def ask_architect(state: dict) -> bool:
    model_name = state.get("architect_model", "gemini-3-flash-preview")
    print(f"📐 [Architect] 規劃系統架構中... (Planning architecture...)")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_KEY}"
    
    # 🌟 Gemini 的嚴格結構化輸出 Schema
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "goal": {"type": "STRING", "description": "系統目標與核心邏輯簡述"},
            "files": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "需要建立或修改的檔案路徑清單"},
            "architecture": {"type": "STRING", "description": "架構設計與設計模式說明"},
            "constraints": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "開發限制、資安注意事項與效能要求"}
        },
        "required": ["goal", "files", "architecture", "constraints"]
    }
    
    lessons = load_lessons()
    sys_prompt = f"你是頂尖的軟體架構師。請根據 Context 與 Task 進行系統規劃。\n\n⚠️ 絕對要遵守的歷史教訓：\n{lessons}"
    
    payload = {
        "system_instruction": {"parts": [{"text": sys_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": f"Context:\n{load_context()}\n\nTask: {state['task']}"}]}],
        "generationConfig": {
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_schema": json_schema
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        # 使用 retry_with_backoff 發送請求，增加穩定性
        resp = retry_with_backoff(lambda: requests.post(url, json=payload, headers=headers, timeout=90))
        
        if resp and resp.status_code == 200:
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            state["plan"] = json.loads(raw_text)
            return True
        elif resp:
            # 💡 抓取 API 回傳的真實錯誤碼 (例如 400 格式錯誤, 403 沒權限)
            print(f"❌ [API 錯誤] 狀態碼: {resp.status_code}")
            print(f"❌ [API 回應]: {resp.text}")
        else:
            print("❌ [API 錯誤] 請求完全失敗 (可能網路斷線或超時)")
            
    except Exception as e:
        # 💡 抓取 Python 執行時的崩潰 (例如 JSON 解析錯誤)
        print(f"❌ [程式例外]: {type(e).__name__} - {e}")
        print("🚨 規劃失敗")
        
    return False

    headers = {"Content-Type": "application/json"}
    resp = retry_with_backoff(lambda: requests.post(url, json=payload, headers=headers, timeout=90))
    
    if resp and resp.status_code == 200:
        try:
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            state["plan"] = json.loads(raw_text)
            return True
        except Exception as e:
            print(f"❌ [程式例外]: {type(e).__name__} - {e}")
            print("🚨 規劃失敗")
    else:
        if resp: print(f"⚠️ Architect API 錯誤: {resp.status_code} - {resp.text}")
    return False

def ask_coder(state: dict) -> bool:
    print(f"💻 [Coder] 撰寫程式碼中... (Generating code for {state['target_file']})...")
    lessons = load_lessons()
    
    # 🌟 1. 組裝純文字的 Prompt (完全放棄 JSON 格式要求，降低模型編碼負擔)
    prompt = f"""你是一個頂尖的 Python Coder。
請根據以下的架構計畫，生成指定的檔案程式碼。

⚠️ 絕對要遵守的歷史教訓：
{lessons}

🎯 目標檔案: {state['target_file']}

📋 架構計畫:
{json.dumps(state['plan'], ensure_ascii=False, indent=2)}

🚨 錯誤回饋 (若有，請務必修正):
{state['errors']}

請直接輸出完整 Python 程式碼，並使用 ```python 和 ``` 標籤包裝。不需要任何額外的解釋。"""

    # 🌟 2. 依照官方 Codex 呼叫方式：input 為單一純字串，並開啟 reasoning
    data = {
        "model": "gpt-5-codex",
        "input": prompt,
        "reasoning": { "effort": "high" }
    }
    
    try:
        # 🌟 3. 確保加上 timeout=300，並呼叫專屬的 v1/responses 端點
        resp = retry_with_backoff(lambda: requests.post("https://api.openai.com/v1/responses", json=data, headers=HEADERS_OPENAI, timeout=300))
        
        if resp and resp.status_code == 200:
            resp_json = resp.json()
            raw_text = ""
            
            # 依照 v1/responses 的結構提取文字
            for item in resp_json.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            raw_text = c.get("text", "")
            
            # 🌟 4. 使用正則表達式精準提取 ```python ... ``` 裡面的純程式碼
            match = re.search(r'```(?:python)?\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                state["code"] = match.group(1).strip()
            else:
                # 如果模型很調皮沒有加上標籤，直接濾掉反引號當作純程式碼
                state["code"] = raw_text.replace("```", "").strip()
                
            return bool(state["code"])
            
        elif resp:
            print(f"❌ [Coder API 錯誤] 狀態碼: {resp.status_code}")
            print(f"❌ [Coder API 回應]: {resp.text}")
        else:
            print("❌ [Coder API 錯誤] 請求完全失敗 (網路斷線或超時)")
            
    except Exception as e:
        print(f"❌ [Coder 執行例外]: {type(e).__name__} - {e}")
        if 'raw_text' in locals():
            print(f"📄 原始回傳內容: {raw_text[:200]}...")
            
    return False

def ask_reviewer(state: dict) -> bool:
    print(f"🔍 [Reviewer] 審查邏輯與安全性... (Auditing code...)")
    sys_prompt = "你是嚴格的安全與代碼審查員。檢查程式碼是否符合邏輯，以及是否包含高風險操作 (如刪除檔案、執行未知名令等)。"
    user_content = f"Task: {state['task']}\n\nCode to review:\n{state['code']}"
    
    data = {
        "model": "gpt-5-codex",
        # 🌟 同樣移除 format 參數
        "input": [
            {"role": "system", "content": sys_prompt + "\n請務必回傳純 JSON，不要包含 Markdown 語法 (如 ```json)。格式為: {\"status\": \"PASS\"/\"FAIL\", \"risk_level\": \"LOW\"/\"MEDIUM\"/\"HIGH\", \"critical_issues\": [\"說明\"], \"summary\": \"總結\"}"},
            {"role": "user", "content": user_content}
        ]
    }
    
    try:
        resp = retry_with_backoff(lambda: requests.post("https://api.openai.com/v1/responses", json=data, headers=HEADERS_OPENAI, timeout=120))
        
        if resp and resp.status_code == 200:
            resp_json = resp.json()
            raw_text = ""
            for item in resp_json.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            raw_text = c.get("text", "")
            
            # 🌟 防彈 JSON 萃取
            clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
            clean_text = re.sub(r"\s*```$", "", clean_text)
            
            state["review"] = json.loads(clean_text)
            return True
        elif resp:
            print(f"❌ [Reviewer API 錯誤] 狀態碼: {resp.status_code}")
        else:
            print("❌ [Reviewer API 錯誤] 請求完全失敗")
            
    except Exception as e:
        print(f"❌ [Reviewer 執行例外]: {type(e).__name__} - {e}")
        if 'raw_text' in locals():
            print(f"📄 原始回傳內容: {raw_text[:200]}...")
            
    return False

def ask_supervisor(state: dict) -> str:
    print("👔 [Supervisor] 判斷錯誤根源中... (Analyzing root cause...)")
    prompt = f"Review 失敗了。計畫: {json.dumps(state['plan'])}\n錯誤: {state['review'].get('critical_issues')}\n請問這是架構計畫的問題，還是工程師沒寫好？請只回答 'architect' 或 'coder'。"
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
    resp = retry_with_backoff(lambda: requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=HEADERS_OPENAI, timeout=60))
    if resp and resp.status_code == 200:
        decision = resp.json()["choices"][0]["message"]["content"].strip().lower()
        if "architect" in decision: return "architect"
    return "coder"

def ask_explainer(state: dict):
    print("📝 [Explainer] 正在為您總結程式碼亮點...")
    prompt = f"請用繁體中文，以 3-5 個條列重點，簡短總結以下 Python 程式碼的核心功能與實作亮點：\n{state['code']}"
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    resp = retry_with_backoff(lambda: requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=HEADERS_OPENAI, timeout=90))
    if resp and resp.status_code == 200:
        state["explanation"] = resp.json()["choices"][0]["message"]["content"].strip()
    else:
        state["explanation"] = "無法生成總結。"

# ------------- 人類審核與儲存 -------------
def stage_for_human_approval(state: dict) -> str:
    # 🌟 將使用者輸入的相對路徑，轉換為相對於「專案根目錄」的絕對路徑
    target = os.path.join(PROJECT_ROOT, state["target_file"])
    staging = target + ".staging"
    
    # 確保目標資料夾存在
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    
    with open(staging, "w", encoding="utf-8") as f:
        f.write(state["code"])
    with open(target + ".approval_log.json", "w", encoding="utf-8") as f:
        json.dump(state["review"], f, indent=2, ensure_ascii=False)

    print(f"\n✅ [Staging] 程式碼已暫存於: {staging}")
    print("\n" + "="*50)
    print("💡 【AI 程式碼功能總結】")
    print(state.get("explanation", "無總結。"))
    print("="*50 + "\n")
    
    # 🌟 增強版選單：加入「紀錄教訓」選項
    while True:
        choice = input(f"🔒 [Action Required] 請問要如何處理這個檔案？(Choose an action):\n  [y] 合併 (Merge to target)\n  [r] 重試 (Retry request)\n  [l] 紀錄教訓 (Log a lesson)\n  [n] 放棄 (Abort)\n👉 請選擇 Select (y/r/l/n): ").strip().lower()
        
        if choice == 'y':
            os.replace(staging, target)
            print(f"🎉 成功！已合併至 {target}")
            return "merged"
        elif choice == 'r':
            return "retry"
        elif choice == 'l':
            lesson = input("📝 請輸入要讓系統記住的教訓 (例如：FastAPI 必須使用 Pydantic)：").strip()
            if lesson:
                save_lesson(lesson)
                print(f"✅ 已將教訓寫入錯題本：{lesson}")
            # 不 return，讓用戶寫完教訓後可以繼續選擇要合併還是重試
        elif choice == 'n':
            print("🛑 使用者放棄本次變更。")
            return "aborted"
        else:
            print("❌ 無效輸入。")

# ------------- 主流程 Graph State Machine -------------
def get_task_from_context() -> str:
    """自動從 context.md 擷取 Current Task 區塊的內容"""
    context = load_context()
    # 使用 Regex 尋找 "Current Task" 標題下的所有內容，直到遇到下一個 "##" 或檔案結束
    match = re.search(r"##\s*\d*\.?\s*Current Task[^\n]*\n(.*?)(?:##|$)", context, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "無法從 context.md 找到 Current Task 區塊，請確認文件格式。"

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 [Init] 歡迎進入自動開發模式 / Welcome to Auto-Dev Mode")
    print("="*50)
    
    # 🌟 自動從 context.md 抓取任務
    auto_task = get_task_from_context()
    print("\n📝 [Current Task] 讀取當前任務 / Loaded task from context.md:")
    print("-" * 40)
    print(auto_task)
    print("-" * 40 + "\n")

    # 嘗試讓用戶輸入，如果 CMD 跳過或用戶直接按 Enter，就使用自動抓取的任務
    try:
        user_task = input("👉 Manual input (Enter to use context task): ").strip()
    except EOFError:
        user_task = ""
        
    if not user_task:
        user_task = auto_task

    # 🌟 智慧路徑預測：自動從任務內容尋找 .py 檔名
    default_path = "api.py" # 預設最後防線
    path_match = re.search(r"`?([a-zA-Z0-9_/\\]+\.py)`?", user_task)
    if path_match:
        default_path = path_match.group(1)

    try:
        user_target = input(f"🎯 [Input] 請輸入目標檔案路徑 / Enter target file path (預設 / Default: {default_path}): ").strip()
    except EOFError:
        user_target = ""
        
    if not user_target:
        user_target = default_path

    # 初始化全域狀態字典
    STATE = {
        "task": user_task,
        "target_file": user_target,
        "plan": None,
        "code": None,
        "review": None,
        "errors": [],
        "explanation": None,
        "architect_model": "gemini-3-flash-preview"
    }

    print("\n請選擇本次任務的架構規劃複雜度：")
    print("  [1] ⚡ 一般任務 (gemini-3-flash-preview)")
    print("  [2] 🧠 複雜任務 (gemini-2.5-pro)")
    
    try:
        choice = input("👉 請選擇 (1 或 2，直接 Enter 預設為 1): ").strip()
    except EOFError:
        choice = ""
        
    if choice == "2":
        STATE["architect_model"] = "gemini-2.5-pro"
    
    print(f"\n✅ 任務啟動：目標檔案 [{STATE['target_file']}]")
    print(f"✅ 使用模型：{STATE['architect_model']}\n")

    user_satisfied = False

    while not user_satisfied:
        STATE["errors"] = []
        
        # 節點 1: Architect
        if not STATE["plan"]:
            if not ask_architect(STATE): raise SystemExit("🚨 規劃失敗")
        
        retry_count = 0
        while retry_count < 3:
            print(f"\n--- 🔄 開發循環 Iteration {retry_count+1}/3 ---")
            
            # 節點 2: Coder
            if not ask_coder(STATE): raise SystemExit("🚨 編碼失敗")
            
            # 節點 3: 靜態檢查
            syntax_err = safe_ast_check(STATE["code"])
            danger = dangerous_pattern_scan(STATE["code"])
            if syntax_err or danger:
                STATE["errors"] = [f"Syntax: {syntax_err}", f"Danger: {danger}"]
                retry_count += 1
                continue
                
            # 節點 4: Reviewer
            if not ask_reviewer(STATE):
                retry_count += 1
                continue
            
            rev = STATE["review"]
            print(f"📊 [Review Result] Status: {rev.get('status')} | Risk: {rev.get('risk_level')}")
            
            if rev.get("risk_level") == "HIGH":
                raise SystemExit(f"🚨 高風險代碼，強制終止: {rev.get('critical_issues')}")
            
            if rev.get("status") == "FAIL":
                STATE["errors"] = rev.get("critical_issues", ["Unknown"])
                next_node = ask_supervisor(STATE)
                print(f"👔 Supervisor 決定退回給: {next_node}")
                if next_node == "architect":
                    STATE["plan"] = None
                    break
                else:
                    retry_count += 1
                    continue

            break
            
        if not STATE["code"] or STATE["review"].get("status") != "PASS":
            raise SystemExit("🚨 達到最大重試次數，任務失敗。")

        # 節點 5: Explainer & Human
        ask_explainer(STATE)
        action = stage_for_human_approval(STATE)
        
        if action in ("merged", "aborted"):
            user_satisfied = True
        elif action == "retry":
            print("\n🚀 重新啟動整個開發流程...\n")
            STATE["plan"] = None