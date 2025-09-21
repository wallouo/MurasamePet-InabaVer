# -*- coding: utf-8 -*-
"""
FastAPI 後端 API：聊天、雙語回覆、語音合成、摸頭。
- 優先 VOICEVOX；不可用時改用 mock TTS（正弦波）。
- 嚴格檢查返回 wav（避免“有檔無聲”）。
"""

from __future__ import annotations
import hashlib
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------------------- 環境變數 --------------------
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT', 'http://127.0.0.1:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:8b')

VOICEVOX_ENDPOINT = os.getenv('VOICEVOX_ENDPOINT', 'http://127.0.0.1:50021')
VOICEVOX_SPEAKER = int(os.getenv('VOICEVOX_SPEAKER', '1'))
TTS_BACKEND = os.getenv('TTS_BACKEND', 'voicevox').lower()

voices_dir = Path('voices')
voices_dir.mkdir(exist_ok=True)

# -------------------- 應用 --------------------
app = FastAPI()


# -------------------- 工具：mock wav --------------------
def _ensure_voices_dir() -> None:
    voices_dir.mkdir(exist_ok=True)


def generate_beep_wav(path: Path, duration: float = 1.2, freq: float = 660.0, sr: int = 24000, **kwargs):
    """生成一段正弦波 wav（mock）。兼容 seconds/duration 兩種參數名。"""
    # 兼容以前寫法：seconds=...
    if 'seconds' in kwargs and isinstance(kwargs['seconds'], (int, float)):
        duration = float(kwargs['seconds'])

    _ensure_voices_dir()
    nframes = int(sr * duration)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(nframes):
            val = int(32767 * 0.25 * math.sin(2 * math.pi * freq * i / sr))
            wf.writeframes(struct.pack('<h', val))


# -------------------- VOICEVOX 調用 --------------------
def voicevox_available() -> bool:
    try:
        r = requests.get(f"{VOICEVOX_ENDPOINT}/version", timeout=3)
        return r.ok
    except Exception:
        return False


def call_voicevox_wav_path(ja_text: str) -> Optional[str]:
    """
    調用 VOICEVOX：audio_query -> synthesis
    - 自動補強常用參數（避免音量 0 或過短無聲）
    - 成功回 wav 絕對路徑，失敗回 None
    """
    try:
        # 1) audio_query
        r1 = requests.post(
            f"{VOICEVOX_ENDPOINT}/audio_query",
            params={"text": ja_text, "speaker": VOICEVOX_SPEAKER},
            timeout=10
        )
        if not r1.ok:
            return None
        query = r1.json()

        # ---- 補強常用參數（保險）----
        query["volumeScale"] = max(0.8, float(query.get("volumeScale", 1.0)))
        query["intonationScale"] = float(query.get("intonationScale", 1.0))
        query["speedScale"] = float(query.get("speedScale", 1.0))
        query["pitchScale"] = float(query.get("pitchScale", 0.0))
        query["prePhonemeLength"] = float(query.get("prePhonemeLength", 0.1))
        query["postPhonemeLength"] = float(query.get("postPhonemeLength", 0.1))

        # 2) synthesis（**一定用 JSON 字串**）
        r2 = requests.post(
            f"{VOICEVOX_ENDPOINT}/synthesis",
            params={"speaker": VOICEVOX_SPEAKER, "enable_interrogative_upspeak": "true"},
            headers={"Content-Type": "application/json"},
            data=json.dumps(query),
            timeout=30
        )
        if not r2.ok:
            return None

        # 寫檔（md5 以文字內容生成）
        md5 = hashlib.md5(ja_text.encode('utf-8')).hexdigest()
        wav_path = (voices_dir / f"{md5}.wav").resolve()
        with open(wav_path, 'wb') as f:
            f.write(r2.content)

        # 檔案過小視為失敗（常見：無聲或例外卻 200）
        if wav_path.stat().st_size < 20480:  # 20KB
            return None

        return str(wav_path)
    except Exception:
        return None


# -------------------- 模型/聊天 --------------------
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]


@app.post('/qwen3')
async def qwen3(req: ChatRequest) -> Dict[str, Any]:
    """代理 Ollama 的聊天接口；失敗則 echo 最後一條 user。"""
    messages = req.messages or []
    try:
        resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        reply = None
        if isinstance(data, dict):
            if 'message' in data and isinstance(data['message'], dict):
                reply = data['message'].get('content')
            elif 'choices' in data and data['choices']:
                reply = data['choices'][0]['message']['content']
        if not reply:
            raise RuntimeError('no reply')
        history = messages + [{"role": "assistant", "content": reply}]
        return {"response": reply, "history": history}
    except Exception:
        last = messages[-1]['content'] if messages else ''
        history = messages + [{"role": "assistant", "content": last}]
        return {"response": last, "history": history}


# -------------------- 雙語回覆（佔位） --------------------
class ReplyBiRequest(BaseModel):
    text: Optional[str] = None
    zh: Optional[str] = None
    ja: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None


@app.post('/reply_bi')
async def reply_bi(req: ReplyBiRequest) -> Dict[str, Any]:
    """
    生成中日雙語回覆（暫時簡化：不翻譯，複用輸入）。
    之後可替換為真正的雙語生成/翻譯。
    """
    text = (req.text or req.zh or req.ja or '').strip()
    zh = (req.zh or text).strip()
    ja = (req.ja or text).strip()
    history = req.history or []
    history.append({"role": "assistant", "content": ja})
    return {"zh": zh, "ja": ja, "history": history}


# -------------------- TTS --------------------
class TTSRequest(BaseModel):
    ja: str
    zh: Optional[str] = None  # 可選：讓 /tts 也能把字幕回前端


@app.post('/tts')
async def tts(req: TTSRequest) -> Dict[str, Any]:
    """
    語音合成：優先 VOICEVOX，失敗/mock 回退。
    回傳: {"wav_path": "...", "subtitle_zh": "...", "backend": "voicevox|mock|cache", "error": "...(可選)"}
    """
    try:
        ja_text = (req.ja or "").strip()
        subtitle_zh = (req.zh or "").strip()
        if not ja_text:
            raise HTTPException(status_code=400, detail="ja text is empty")

        _ensure_voices_dir()
        md5 = hashlib.md5(ja_text.encode('utf-8')).hexdigest()
        wav_file = (voices_dir / f"{md5}.wav").resolve()

        # 命中快取（>20KB 視為正常語音）
        if wav_file.exists() and wav_file.stat().st_size >= 20480:
            return {"wav_path": str(wav_file), "subtitle_zh": subtitle_zh, "backend": "cache"}

        # 嘗試 VOICEVOX
        if TTS_BACKEND == 'voicevox' and voicevox_available():
            wav_path = call_voicevox_wav_path(ja_text)
            if wav_path:
                return {"wav_path": wav_path, "subtitle_zh": subtitle_zh, "backend": "voicevox"}

        # 走 mock
        mock_file = (voices_dir / f"{md5}_mock.wav").resolve()
        generate_beep_wav(mock_file, seconds=1.2, freq=660.0)
        return {"wav_path": str(mock_file), "subtitle_zh": subtitle_zh, "backend": "mock"}

    except HTTPException:
        raise
    except Exception as e:
        # 最終保底：永不 500，回 mock 並帶錯誤訊息
        try:
            _ensure_voices_dir()
            md5 = hashlib.md5((req.ja or "mock").encode('utf-8')).hexdigest()
            mock_file = (voices_dir / f"{md5}_mock.wav").resolve()
            generate_beep_wav(mock_file, seconds=1.2, freq=660.0)
            return {
                "wav_path": str(mock_file),
                "subtitle_zh": (req.zh or "") if hasattr(req, 'zh') else "",
                "backend": "mock",
                "error": f"{type(e).__name__}: {e}"
            }
        except Exception as e2:
            # 真的連 mock 都失敗才回 500（極少）
            raise HTTPException(status_code=500, detail=f"tts fatal: {type(e2).__name__}: {e2}")


# -------------------- 一條龍 /say --------------------
class SayRequest(BaseModel):
    text: Optional[str] = None
    zh: Optional[str] = None
    ja: Optional[str] = None


@app.post('/say')
async def say(req: SayRequest) -> Dict[str, Any]:
    """
    入口：輸入 {text} 或 {zh, ja}
    - 若只有 text：用 /qwen3 生成一句；ja=回覆，zh=同文（暫時）
    - TTS 合成，回 wav 路徑與字幕
    """
    zh = (req.zh or "").strip()
    ja = (req.ja or "").strip()

    if not ja:
        text = (req.text or "").strip()
        # 用聊天生一段（最短回覆）
        if not text:
            text = "テストです"
        messages = [{"role": "user", "content": text}]
        chat_res = await qwen3(ChatRequest(messages=messages))
        ja = chat_res.get("response", "テストです")
        if not zh:
            zh = ja

    # 調 TTS
    tts_res = await tts(TTSRequest(ja=ja, zh=zh))
    return {
        "wav_path": tts_res["wav_path"],
        "subtitle_zh": zh or ja,
        "backend": tts_res.get("backend", "unknown")
    }


# -------------------- 摸頭 /pat --------------------
@app.post('/pat')
async def pat() -> Dict[str, Any]:
    """
    摸頭端點：觸發一句短台詞並合成語音。
    之後可改為讀 persona 或模板。
    """
    return await say(SayRequest(text="頭をなでる"))
