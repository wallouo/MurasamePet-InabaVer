import os
import re
import time
import random
import json
import requests
import ast
import threading
from typing import Optional, Dict, Any

# --- 環境變數 ---
PPLX_KEY = os.environ.get("PPLX_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

if not PPLX_KEY or not OPENAI_KEY:
    raise SystemExit("❌ 找不到 API Key。請確認 PPLX_API_KEY 與 OPENAI_API_KEY 已設定。")

HEADERS_PPLX = {"Authorization": f"Bearer {PPLX_KEY}", "Content-Type": "application/json"}
HEADERS_OPENAI = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}

LOCK = threading.RLock()

# ------------- 基礎工具 -------------

def retry_with_backoff(call_fn, tries=5, base=1.0, max_delay=30.0):
    for attempt in range(1, tries + 1):
        resp = call_fn()
        if resp is None:
            return None
        if resp.status_code in (200, 201):
            return resp
        if resp.status_code in (429, 502, 503, 504):
            wait = min(max_delay, base * (2 ** (attempt - 1)) * (1 + random.random() * 0.1))
            print(f"Rate/Server error {resp.status_code}, backing off {wait:.1f}s (attempt {attempt}/{tries})")
            time.sleep(wait)
            continue
        print("Non-retriable error:", resp.status_code, resp.text[:300])
        return resp
    print("Exceeded API retries")
    return None

def load_context() -> str:
    try:
        with open("context.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def extract_json_block(text: str) -> dict:
    """強化版 JSON 解析，處理 LLM 可能加上 ```json ``` 標籤的情況"""
    try:
        m = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.S)
        raw_str = m.group(1).strip() if m else text.strip()
        return json.loads(raw_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}\nRaw output: {text}")

def safe_ast_check(code: str) -> Optional[str]:
    try:
        ast.parse(code)
        return None
    except Exception as e:
        return str(e)

def dangerous_pattern_scan(code: str) -> list:
    """已加入 rm -rf 等系統破壞性指令的嚴格靜態掃描"""
    patterns = [
        r"\beval\s*\(", r"\bexec\s*\(", r"subprocess\.Popen", 
        r"os\.system", r"requests\.post\(", 
        r"rm\s+-rf", r"shutil\.rmtree", r"os\.remove", # 新增刪除檔案的高危指令
        r"OPENAI_API_KEY", r"PPLX_API_KEY"
    ]
    found = [p for p in patterns if re.search(p, code)]
    return found

# ------------- Agent 介面 (嚴格遵循 v2 Schema) -------------

def ask_architect(task: str) -> Optional[Dict[str, Any]]:
    print(f"📐 [Architect] Planning task: {task}")
    prompt = (
        "You are a Lead Software Architect. Provide a detailed implementation plan.\n"
        "You MUST output STRICTLY JSON matching this schema:\n"
        "{\n  \"goal\": \"...\",\n  \"files\": [\"file1.py\", ...],\n  \"architecture\": \"...\",\n  \"constraints\": [\"...\"]\n}\n"
    )
    data = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context:\n{load_context()}\n\nTask: {task}"}
        ]
    }
    resp = retry_with_backoff(lambda: requests.post("https://api.perplexity.ai/chat/completions", json=data, headers=HEADERS_PPLX, timeout=30))
    if resp and resp.status_code == 200:
        try:
            return extract_json_block(resp.json()["choices"][0]["message"]["content"])
        except ValueError as e:
            print(f"Architect JSON parsing failed: {e}")
    return None

def ask_coder(plan: Dict, target_file: str, feedback: list = None) -> Optional[str]:
    print(f"💻 [Coder] Writing code for {target_file}...")
    
    sys_prompt = (
        "You are an Expert Python Developer. Follow the architect's plan strictly. "
        "No hidden side effects. No shell/system execution.\n"
        "Output STRICTLY JSON matching this schema:\n"
        "{\n  \"files\": {\n    \"filename.py\": \"code...\"\n  }\n}"
    )
    
    user_content = f"Plan:\n{json.dumps(plan, indent=2)}\n\nWrite code for: {target_file}."
    if feedback:
        print(f"⚠️ [Coder] Applying feedback from previous failure: {feedback}")
        user_content += f"\n\nCRITICAL ISSUES TO FIX from previous review:\n{json.dumps(feedback, indent=2)}"

    data = {
        "model": "gpt-5.2-codex", # 替換為你實際使用的 OpenAI 模型 (如 gpt-4o)
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
    }
    resp = retry_with_backoff(lambda: requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=HEADERS_OPENAI, timeout=60))
    if resp and resp.status_code == 200:
        try:
            result = extract_json_block(resp.json()["choices"][0]["message"]["content"])
            # 提取實際程式碼
            return result.get("files", {}).get(target_file) or list(result.get("files", {}).values())[0]
        except Exception as e:
            print(f"Coder JSON parsing failed: {e}")
    return None

def ask_reviewer(code: str, plan: Dict, strict_mode: bool = False) -> Optional[Dict[str, Any]]:
    print("🔎 [Reviewer] Auditing generated code...")
    sys_prompt = (
        "You are a Senior Security & Code Reviewer. Perform static analysis, logic validation, and detect unsafe operations.\n"
        "You MUST output STRICTLY JSON. NEVER output free text outside JSON.\n"
        "Schema:\n"
        "{\n  \"status\": \"PASS\" or \"FAIL\",\n  \"critical_issues\": [\"issue1\"], (empty if PASS)\n  \"risk_level\": \"LOW\" or \"MEDIUM\" or \"HIGH\",\n  \"summary\": \"...\"\n}\n"
    )
    if strict_mode:
         sys_prompt += "\nWARNING: Previous output was invalid JSON. You MUST output ONLY valid JSON this time."

    data = {
        "model": "gpt-5.2-codex",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Architect Plan:\n{json.dumps(plan)}\n\nCode to review:\n{code}"}
        ]
    }
    resp = retry_with_backoff(lambda: requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=HEADERS_OPENAI, timeout=30))
    if resp and resp.status_code == 200:
        try:
            return extract_json_block(resp.json()["choices"][0]["message"]["content"])
        except ValueError:
            return None # 解析失敗回傳 None，觸發重試
    return None

# ------------- 執行與佈署 -------------

def stage_for_human_approval(target_file: str, code: str, review: dict):
    """遵守 Human-Gated 原則：不直接覆蓋檔案，僅產生 staging 檔並紀錄"""
    staging = target_file + ".staging"
    os.makedirs(os.path.dirname(target_file) or ".", exist_ok=True)
    with open(staging, "w", encoding="utf-8") as f:
        f.write(code)
    
    log_file = target_file + ".approval_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dumps(review, indent=2)

    print(f"\n✅ [Human-Gated Execution] Code verified and staged at: {staging}")
    print(f"🔒 Awaiting manual approval via Dashboard/OpenClaw to merge into {target_file}.")

# ------------- 主流程 Orchestrator -------------

if __name__ == "__main__":
    TASK = "Create a module 'logic/memory.py' that handles saving and loading user memory (name, last_interaction) to a JSON file."
    target = "logic/memory.py"

    # 1. 架構師階段
    plan = ask_architect(TASK)
    if not plan or "goal" not in plan:
        raise SystemExit("🚨 HALT: Architect failed to generate a valid plan JSON.")

    MAX_RETRY = 2
    retry_count = 0
    critical_issues = None
    final_approved_code = None
    final_review = None

    # 2. 狀態機與重試迴圈
    while retry_count <= MAX_RETRY:
        print(f"\n--- 🔄 Iteration {retry_count} / {MAX_RETRY} ---")
        
        # Coder 產出程式碼
        code = ask_coder(plan, target, feedback=critical_issues)
        if not code:
            raise SystemExit("🚨 HALT: Coder failed to generate code.")

        # 本地靜態安全檢查 (Fail-Closed: 絕不送 Reviewer)
        syntax_err = safe_ast_check(code)
        danger = dangerous_pattern_scan(code)
        
        if syntax_err or danger:
            print(f"⚠️ [Static Analysis Failed] Syntax: {syntax_err}, Danger: {danger}")
            critical_issues = [f"Syntax Error: {syntax_err}", f"Dangerous Pattern: {danger}"]
            retry_count += 1
            continue

        # Reviewer 審查
        review = ask_reviewer(code, plan, strict_mode=(retry_count>0))
        
        # 檢查 1: JSON 格式無效或缺漏
        if not review or "status" not in review or "risk_level" not in review:
            print("⚠️ [Reviewer] Invalid JSON or missing required fields.")
            retry_count += 1
            continue

        print(f"📊 [Review Result] Status: {review['status']} | Risk: {review['risk_level']}")
        
        # 檢查 2: 高風險直接停機 (Fail-Closed)
        if review["risk_level"] == "HIGH":
            raise SystemExit(f"🚨 HALT: Reviewer flagged code as HIGH RISK. Issues: {review.get('critical_issues')}")

        # 檢查 3: 邏輯或安全不通過，退回給 Coder
        if review["status"] == "FAIL":
            print(f"❌ [Reviewer FAILED] Issues: {review.get('critical_issues')}")
            critical_issues = review.get("critical_issues", ["Unknown failure reason"])
            retry_count += 1
            continue

        # 檢查 4: 通過且風險在可控範圍
        if review["status"] == "PASS" and review["risk_level"] in ["LOW", "MEDIUM"]:
            final_approved_code = code
            final_review = review
            break # 成功跳出迴圈

    # 3. 迴圈結束判定
    if not final_approved_code:
        raise SystemExit("🚨 HALT: Max retries reached or conditions not met. Pipeline aborted.")

    # 4. 人類授權執行
    stage_for_human_approval(target, final_approved_code, final_review)