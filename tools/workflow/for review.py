import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx
import psutil

try:
    import pygetwindow as gw  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gw = None

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL_NAME = "qwen"


SYSTEM_PROCESS_DENYLIST = {
    "system",
    "system idle process",
    "idle",
    "registry",
    "memcompression",
    "dwm.exe",
    "lsass.exe",
    "svchost.exe",
    "fontdrvhost.exe",
    "ctfmon.exe",
    "smss.exe",
    "wininit.exe",
    "services.exe",
    "searchindexer.exe",
    "shellexperiencehost.exe",
    "taskhostw.exe",
    "taskmgr.exe",
    "explorer.exe",
    "cmd.exe",
    "conhost.exe",
}


UNKNOWN_RESULT: Dict[str, Any] = {
    "activity_type": "Unknown",
    "confidence": 0.0,
    "reason": "LLM analysis failed or unavailable.",
}


@dataclass
class AppWatcher:
    """Collect process/window context and delegate activity reasoning to an LLM."""
    model: str = DEFAULT_MODEL_NAME
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    request_timeout: float = 30.0
    max_processes: int = 40
    max_windows: int = 15
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False, repr=False)

    async def infer_activity(self) -> Dict[str, Any]:
        processes = self.collect_process_snapshot()
        windows = self.collect_window_titles()
        return await analyze_activity_with_llm(
            processes,
            windows,
            model=self.model,
            endpoint=self.endpoint,
            timeout=self.request_timeout,
        )

    def collect_process_snapshot(self) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        seen_names: set[str] = set()

        for proc in psutil.process_iter(
            attrs=["pid", "name", "username", "cmdline", "exe"]
        ):
            info = proc.info
            name = (info.get("name") or "").strip()
            if not name:
                continue
            normalized = name.lower()
            if normalized in SYSTEM_PROCESS_DENYLIST:
                continue
            if normalized in seen_names:
                continue
            seen_names.add(normalized)

            entry: Dict[str, Any] = {
                "pid": info.get("pid"),
                "name": name,
                "username": info.get("username"),
            }

            cmdline = info.get("cmdline") or []
            if cmdline:
                entry["cmdline"] = " ".join(cmdline)[:512]

            exe = info.get("exe")
            if exe:
                entry["exe"] = exe

            snapshot.append(entry)
            if len(snapshot) >= self.max_processes:
                break

        return snapshot

    def collect_window_titles(self) -> List[str]:
        if gw is None:
            return []

        titles: List[str] = []
        try:
            for title in gw.getAllTitles():  # type: ignore[attr-defined]
                normalized = title.strip()
                if not normalized:
                    continue
                titles.append(normalized)
                if len(titles) >= self.max_windows:
                    break
        except Exception as exc:  # pragma: no cover - GUI environment specific
            logger.debug("Failed to collect window titles: %s", exc)

        return titles


def _format_processes(process_list: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for proc in process_list:
        parts = [f"{proc.get('name')} (PID {proc.get('pid')})"]
        if proc.get("cmdline"):
            parts.append(f"cmd: {proc['cmdline']}")
        if proc.get("exe"):
            parts.append(f"path: {proc['exe']}")
        if proc.get("username"):
            parts.append(f"user: {proc['username']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines) if lines else "No notable processes detected."


def _format_windows(window_titles: Sequence[str]) -> str:
    return "\n".join(f"- {title}" for title in window_titles) or "No active windows detected."


def _build_prompt(process_list: Sequence[Dict[str, Any]], window_titles: Sequence[str]) -> str:
    browser_hint = (
        "Treat known browser executables (chrome.exe, msedge.exe, firefox.exe, brave.exe, "
        "arc.exe, opera.exe, vivaldi.exe) as potential browsing contexts. When window titles "
        "contain developer tools, GitHub, StackOverflow, or AI/copilot/chatbot references, "
        "recognize these as coding-related research."
    )
    notepad_rule = (
        "If Notepad or similar plain-text editor is active simultaneously with evidence of AI "
        "assistants or chatbot interactions in a browser (e.g., ChatGPT, Copilot, Claude, Gemini), "
        "classify the overall activity as Coding due to code experimentation or prompt engineering."
    )
    steam_rule = (
        "Steam.exe alone, especially without an active Steam or game window title, should not be "
        "interpreted as Gaming; treat it as background unless a specific game window is foreground."
    )
    json_schema = json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["activity_type", "confidence", "reason"],
            "properties": {
                "activity_type": {
                    "type": "string",
                    "enum": [
                        "Coding",
                        "Writing",
                        "Browsing",
                        "Gaming",
                        "Entertainment",
                        "Communication",
                        "Productive",
                        "Idle",
                        "Unknown",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
        },
        indent=2,
    )

    prompt = (
        "You are an observant digital activity analyst. Evaluate the user's current activity "
        "based on running processes and visible window titles. Avoid assumptions not grounded "
        "in the evidence.\n\n"
        f"{browser_hint}\n"
        "Differentiate writing vs. coding by noting presence of IDEs, compilers, terminals, "
        "source-code-related filenames, or developer tooling. Consider communication apps, "
        "media players, and productivity suites accordingly.\n"
        f"{notepad_rule}\n"
        f"{steam_rule}\n"
        "When data is inconclusive, prefer an 'Unknown' activity type with low confidence.\n\n"
        f"JSON Schema (strictly follow this shape):\n{json_schema}\n\n"
        "Return ONLY minified JSON compliant with the schema.\n\n"
        "=== Process Snapshot ===\n"
        f"{_format_processes(process_list)}\n\n"
        "=== Window Titles ===\n"
        f"{_format_windows(window_titles)}\n\n"
        "Respond with your best inference."
    )
    return prompt


def _extract_json_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    candidate = raw_text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        sanitized = [
            line
            for line in lines
            if not line.strip().startswith("```") and line.strip().lower() != "json"
        ]
        candidate = "\n".join(sanitized).strip()

    # Attempt direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: extract substring between first '{' and last '}'
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


async def analyze_activity_with_llm(
    process_list: Sequence[Dict[str, Any]],
    window_titles: Sequence[str],
    *,
    model: str = DEFAULT_MODEL_NAME,
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    prompt = _build_prompt(process_list, window_titles)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, json.JSONDecodeError, asyncio.TimeoutError) as exc:
        logger.warning("LLM request failed: %s", exc)
        return UNKNOWN_RESULT.copy()

    raw_output = data.get("response") or data.get("output") or ""
    parsed = _extract_json_payload(raw_output)
    if not isinstance(parsed, dict):
        logger.debug("Failed to parse LLM JSON response: %s", raw_output)
        return UNKNOWN_RESULT.copy()

    if not {"activity_type", "confidence", "reason"} <= parsed.keys():
        logger.debug("LLM response missing required keys: %s", parsed)
        return UNKNOWN_RESULT.copy()

    try:
        parsed["confidence"] = float(parsed["confidence"])
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0

    return parsed