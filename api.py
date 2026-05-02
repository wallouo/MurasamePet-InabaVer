# -*- coding: utf-8 -*-
"""
FastAPI 後端 API：聊天、雙語回覆、語音合成、摸頭。
TTS 策略：
1. 預定義音頻（原聲 Ciallo 等）
2. VITS (vits-simple-api)
3. Mock (保底)
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import struct
import wave
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

# -------------------- 環境變數 --------------------
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
OLLAMA_MODEL = "meguru"

VITS_ENDPOINT = os.getenv('VITS_ENDPOINT', 'http://127.0.0.1:23456')
VITS_SPEAKER_ID = int(os.getenv('VITS_SPEAKER_ID', '88'))

PREDEFINED_AUDIO_DIR = Path(os.getenv(
    'PREDEFINED_AUDIO_DIR',
    'assets/audio'
))

voices_dir = Path('voices')
voices_dir.mkdir(exist_ok=True)

# -------------------- 應用 --------------------
app = FastAPI()

class UserChatRequest(BaseModel):
    text: str
    user_id: str = "master"

@app.post('/chat_process')
async def chat_process(req: UserChatRequest) -> Dict[str, Any]:
    """處理用戶聊天請求，調用 Ollama 並返回 TTS 音頻"""
    user_text = req.text.strip()
    
    # 1. 建構標準 Chat 訊息列表
    # 因為 System Prompt 已在 Modelfile 設定，這裡不需要再傳送 system role
    chat_messages = [
        {"role": "user", "content": user_text}
    ]
    # 2. 準備 Payload
    ollama_payload = {
        "model": OLLAMA_MODEL,
        "messages": chat_messages,
        "stream": False,
        "options": {
            "num_gpu": -1,      # 自動使用 GPU
        }
    }

    print(f"[聊天] 使用者: {user_text}")

    try:
        # 3. 發送請求 (使用標準 /api/chat)
        ollama_resp = requests.post(
            f"{OLLAMA_ENDPOINT}/api/chat",
            json=ollama_payload,
            timeout=60.0 # 給予足夠時間生成
        )
        
        ollama_resp.raise_for_status()
        response_json = ollama_resp.json()
        
        # 4. 解析回應 (Chat API 回傳在 message.content)
        ai_reply_text = response_json.get("message", {}).get("content", "").strip()
        
        # ===== 清洗輸出：檢測非預期的英文（使用白名單機制）=====
        # 允許的英文詞彙白名單
        allowed_english_words = ["ciallo", "Ciallo", "CIALLO"]
        
        # 移除白名單詞彙後檢測
        temp_text = ai_reply_text
        for word in allowed_english_words:
            temp_text = temp_text.replace(word, "")
        
        # 檢測剩餘內容中的英文（連續3個以上英文字母）
        if re.search(r'[a-zA-Z]{3,}', temp_text):
            print(f"[聊天警告] 偵測到非預期的英文輸出: {ai_reply_text}")
            # 如果出現非預期英文，可能是模型錯亂
            # 這裡簡單處理：記錄警告但不阻止輸出
            pass

        if not ai_reply_text:
            ai_reply_text = "えっと...何か言おうとしたんだけど、忘れちゃった♪"

        print(f"[聊天] 巡: {ai_reply_text}")

    except Exception as e:
        print(f"[聊天錯誤] {type(e).__name__}: {e}")
        ai_reply_text = "ごめん、ちょっと考えすぎちゃった..."
    
    # TTS 和返回
    # 這裡假設回覆是純日文，所以中日文欄位都填一樣的
    # 如果未來要做翻譯，可以在這裡呼叫翻譯 API
    subtitle_zh = ai_reply_text 
    
    tts_result = await tts(TTSRequest(ja=ai_reply_text, zh=subtitle_zh))

    return {
        "text": ai_reply_text,
        "subtitle_zh": subtitle_zh,
        "wav_path": tts_result["wav_path"],
        "emotion": "happy",
        "backend": tts_result["backend"]
    }

# -------------------- 雙語回覆 (保留介面) --------------------
class ReplyBiRequest(BaseModel):
    text: Optional[str] = None
    zh: Optional[str] = None
    ja: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None

@app.post('/reply_bi')
async def reply_bi(req: ReplyBiRequest) -> Dict[str, Any]:
    text = (req.text or req.zh or req.ja or '').strip()
    zh = (req.zh or text).strip()
    ja = (req.ja or text).strip()
    history = req.history or []
    history.append({"role": "assistant", "content": ja})
    return {"zh": zh, "ja": ja, "history": history}

# -------------------- TTS --------------------
class TTSRequest(BaseModel):
    ja: str
    zh: Optional[str] = None

@app.post('/tts')
async def tts(req: TTSRequest) -> Dict[str, Any]:
    """
    語音合成優先順序：
    1. 預定義音頻（原聲 Ciallo）+ 後續內容合併
    2. VITS
    3. Mock
    """
    try:
        ja_text = (req.ja or "").strip()
        subtitle_zh = (req.zh or "").strip()

        if not ja_text:
            raise HTTPException(status_code=400, detail="ja text is empty")

        _ensure_voices_dir()
        md5 = hashlib.md5(ja_text.encode('utf-8')).hexdigest()

        # 1. 檢查緩存
        cached_vits = (voices_dir / f"{md5}_vits.wav").resolve()
        cached_merged = (voices_dir / f"{md5}_merged.wav").resolve()
        
        if cached_vits.exists() and cached_vits.stat().st_size >= 10240:
            return {"wav_path": str(cached_vits), "subtitle_zh": subtitle_zh, "backend": "cache"}
        
        if cached_merged.exists() and cached_merged.stat().st_size >= 10240:
            return {"wav_path": str(cached_merged), "subtitle_zh": subtitle_zh, "backend": "cache_merged"}

        # 2. 檢查是否包含打招呼 + 後續內容
        greeting_keyword, remaining_text = split_greeting_text(ja_text)
        
        if greeting_keyword and remaining_text:
            # 有打招呼 + 後續內容，需要合併音頻
            print(f"[語音合成] 偵測到打招呼 '{greeting_keyword}' + 後續內容 '{remaining_text}'")
            
            predefined_path = get_predefined_audio(ja_text)
            
            if predefined_path and os.path.exists(predefined_path) and vits_available():
                # 為後續內容生成 VITS 音頻
                vits_path = call_vits_wav_path(remaining_text)
                
                if vits_path:
                    # 合併音頻
                    merged_path = str(cached_merged)
                    if merge_wav_files(predefined_path, vits_path, merged_path):
                        print(f"[語音合成] 音頻合併成功: 預定義音頻 + VITS")
                        return {"wav_path": merged_path, "subtitle_zh": subtitle_zh, "backend": "predefined+vits"}
                    else:
                        # 合併失敗，只返回預定義音頻
                        return {"wav_path": predefined_path, "subtitle_zh": subtitle_zh, "backend": "predefined_only"}
        
        # 3. 只有打招呼（無後續內容）
        predefined_path = get_predefined_audio(ja_text)
        if predefined_path and os.path.exists(predefined_path):
            return {"wav_path": predefined_path, "subtitle_zh": subtitle_zh, "backend": "predefined"}

        # 4. 嘗試 VITS
        if vits_available():
            vits_path = call_vits_wav_path(ja_text)
            if vits_path:
                return {"wav_path": vits_path, "subtitle_zh": subtitle_zh, "backend": "vits"}

        # 5. Mock 保底
        mock_file = (voices_dir / f"{md5}_mock.wav").resolve()
        generate_beep_wav(mock_file, seconds=1.2, freq=660.0)
        return {"wav_path": str(mock_file), "subtitle_zh": subtitle_zh, "backend": "mock"}

    except HTTPException:
        raise
    except Exception as e:
        # 最終保底
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
            raise HTTPException(status_code=500, detail=f"tts fatal: {type(e2).__name__}: {e2}")

# -------------------- /say --------------------
class SayRequest(BaseModel):
    text: Optional[str] = None
    zh: Optional[str] = None
    ja: Optional[str] = None

@app.post('/say')
async def say(req: SayRequest) -> Dict[str, Any]:
    """入口：輸入 {text} 或 {zh, ja}"""
    zh = (req.zh or "").strip()
    ja = (req.ja or "").strip()

    if not ja:
        text = (req.text or "").strip()
        if not text:
            text = "テストです"
        # 轉發給 chat_process
        chat_res = await chat_process(UserChatRequest(text=text))
        return chat_res

    if not zh:
        zh = ja

    tts_res = await tts(TTSRequest(ja=ja, zh=zh))
    return {
        "wav_path": tts_res["wav_path"],
        "subtitle_zh": zh or ja,
        "backend": tts_res.get("backend", "unknown")
    }

# -------------------- /pat --------------------
@app.post('/pat')
async def pat() -> Dict[str, Any]:
    """摸头端点 - 让模型自由回应"""
    # 给模型一个摸头的上下文，让它自由回应
    return await chat_process(UserChatRequest(
        text="*摸摸头*",  # 模拟摸头动作，让模型做出反应
        user_id="master"
    ))


# -------------------- 工具函式 --------------------

def _ensure_voices_dir() -> None:
    voices_dir.mkdir(exist_ok=True)

def generate_beep_wav(path: Path, duration: float = 1.2, freq: float = 660.0, sr: int = 24000, **kwargs):
    """生成一段正弦波 wav（mock）"""
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

def vits_available() -> bool:
    """檢查 vits-simple-api 是否可用"""
    try:
        r = requests.get(f"{VITS_ENDPOINT}/", timeout=3)
        return r.ok
    except Exception:
        return False

def call_vits_wav_path(ja_text: str) -> Optional[str]:
    """
    呼叫 VITS API 生成語音
    成功返回 wav 絕對路徑，失敗返回 None
    """
    try:
        encoded_text = quote(ja_text)
        url = f"{VITS_ENDPOINT}/voice/vits?text={encoded_text}&id={VITS_SPEAKER_ID}&format=wav&lang=ja"
        print(f"[VITS除錯] 呼叫網址: {url}")
        
        r = requests.get(url, timeout=30)
        
        if not r.ok:
            print(f"[VITS錯誤] 狀態碼 {r.status_code}: {r.text}")
            return None
            
        md5 = hashlib.md5(ja_text.encode('utf-8')).hexdigest()
        wav_path = (voices_dir / f"{md5}_vits.wav").resolve()
        
        with open(wav_path, 'wb') as f:
            f.write(r.content)
            
        file_size = wav_path.stat().st_size
        if file_size < 10240: # 10KB
            print(f"[VITS錯誤] 音頻檔案過小: {file_size} 位元組")
            return None
            
        print(f"[VITS成功] 已生成: {wav_path} ({file_size} 位元組)")
        return str(wav_path)
    except Exception as e:
        print(f"[VITS例外] {type(e).__name__}: {e}")
        return None

def get_predefined_audio(ja_text: str) -> Optional[str]:
    """
    檢查文本開頭是否為打招呼（ciallo等）
    如果是，返回預定義音頻路徑
    """
    predefined_map = {
        "ciallo": "ciallo",
        "ちゃろ": "ciallo",
        "チャロ": "ciallo",
    }
    
    # 清理文本（移除標點和空格）
    clean_text = ja_text.lower().strip()
    for punct in ['～', '！', '!', '？', '?', '。', '、', ' ', '♪']:
        clean_text = clean_text.replace(punct, '')
    
    # 檢查文本開頭是否為打招呼關鍵詞
    for keyword, audio_prefix in predefined_map.items():
        if clean_text.startswith(keyword):  # 改為 startswith，只檢測開頭
            if not PREDEFINED_AUDIO_DIR.exists():
                print(f"[預定義音頻] 目錄不存在: {PREDEFINED_AUDIO_DIR}")
                return None
            candidates = list(PREDEFINED_AUDIO_DIR.glob(f"{audio_prefix}*.wav"))
            if candidates:
                selected = random.choice(candidates)
                print(f"[預定義音頻] 已選擇: {selected.name}")
                return str(selected.resolve())
            else:
                print(f"[預定義音頻] 找不到符合的檔案: {audio_prefix}*.wav")
    return None

def split_greeting_text(ja_text: str) -> tuple[Optional[str], str]:
    """
    分離打招呼和後續文本
    返回: (打招呼關鍵詞, 剩餘文本)
    """
    greetings = ["ciallo", "ちゃろ", "チャロ"]
    
    text_lower = ja_text.lower()
    for greeting in greetings:
        if text_lower.startswith(greeting):
            # 找到打招呼結束的位置
            end_pos = len(greeting)
            # 跳過後續的標點符號
            while end_pos < len(ja_text) and ja_text[end_pos] in ['～', '！', '!', '♪', ' ']:
                end_pos += 1
            
            remaining = ja_text[end_pos:].strip()
            return (greeting, remaining)
    
    return (None, ja_text)

def merge_wav_files(file1: str, file2: str, output: str) -> bool:
    """
    合併兩個 WAV 文件（使用標準庫 wave）
    """
    try:
        # 讀取第一個文件
        with wave.open(file1, 'rb') as w1:
            params1 = w1.getparams()
            frames1 = w1.readframes(w1.getnframes())
        
        # 讀取第二個文件
        with wave.open(file2, 'rb') as w2:
            params2 = w2.getparams()
            frames2 = w2.readframes(w2.getnframes())
        
        # 檢查參數是否兼容
        if params1[:3] != params2[:3]:  # nchannels, sampwidth, framerate
            print(f"[音頻合併] 警告: 音頻參數不同")
            # 如果參數不同，返回第一個文件
            return False
        
        # 合併並寫入新文件
        with wave.open(output, 'wb') as wout:
            wout.setparams(params1)
            wout.writeframes(frames1 + frames2)
        
        print(f"[音頻合併] 成功: {output}")
        return True
        
    except Exception as e:
        print(f"[音頻合併] 失敗: {e}")
        return False