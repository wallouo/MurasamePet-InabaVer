import ast
import json
import re

# 補齊破壞性與高危險指令
DANGEROUS = [
    r"\beval\s*\(", r"\bexec\s*\(", r"subprocess", r"os\.system", 
    r"requests\.post\(", r"rm\s+-rf", r"shutil\.rmtree", r"os\.remove",
    r"OPENAI_API_KEY", r"PPLX_API_KEY"
]

def quick_review(code: str) -> dict:
    critical_issues = []
    
    # 1) Syntax Check
    try:
        ast.parse(code)
    except Exception as e:
        critical_issues.append(f"SyntaxError: {e}")

    # 2) Dangerous patterns Check
    found = []
    for p in DANGEROUS:
        if re.search(p, code):
            found.append(p)
    if found:
        critical_issues.append("Dangerous patterns found: " + ", ".join(found))

    # 3) Risk Scoring & Decision Mapping to v2 Schema
    if found or critical_issues:
        risk_level = "HIGH"
        status = "FAIL"
        summary = "Critical issues or dangerous patterns detected. Do not deploy."
    else:
        risk_level = "LOW"
        status = "PASS"
        summary = "No immediate issues detected. Code passed static quick review."

    # 嚴格遵循 v2 Pipeline Schema
    return {
        "status": status,
        "critical_issues": critical_issues,
        "risk_level": risk_level,
        "summary": summary
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reviewer_agent.py path/to/generated_code.py")
        raise SystemExit(1)
    
    p = sys.argv[1]
    with open(p, "r", encoding="utf-8") as f:
        c = f.read()
    
    # 確保輸出純粹的 JSON，不包含多餘文字
    print(json.dumps(quick_review(c), ensure_ascii=False, indent=2))