<div align="center">

# InabaPet

**A desktop AI companion powered by local LLMs — chat, vision, voice, and head-pats included.**

[![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)](https://github.com/wallouo/InabaPet/releases)
[![Status](https://img.shields.io/badge/status-stable-brightgreen?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-41CD52?style=flat-square&logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com)
[![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](#)
[![Platform](https://img.shields.io/badge/platform-Windows-0078D6?style=flat-square&logo=windows&logoColor=white)](#)

[English](#english) · [中文說明](#中文說明)

</div>

---

## English

A clean, refactored fork of [MurasamePet](https://github.com/LemonQu-GIT/MurasamePet) featuring a **PyQt5** desktop frontend, local LLM chat via Ollama, vision recognition, multilingual TTS, and a fully documented FastAPI backend — all running on-device with no cloud dependency.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | PyQt5 | Desktop pet GUI, sprite rendering, mouse interaction |
| **Backend API** | FastAPI + Uvicorn | REST endpoints, prompt orchestration |
| **Chat Model** | `meguru` (Qwen 3.5 4B fine-tune) | Roleplay, conversation, memory injection |
| **Vision Model** | `qwen3-vl:4b-instruct` | Image analysis via `VisionConnector` |
| **Translation** | `qwen3.5:2b` | ZH ↔ EN ↔ JA via `translate.py` |
| **Speech Synthesis** | VITS (`vits-simple-api`) + Mock fallback | TTS audio generation |
| **Runtime** | Ollama | Local model serving |
| **Scripting** | PowerShell (`run_local.ps1`) | One-click environment setup & launch |

---

## AI Models

| Role | Model | Status |
|---|---|---|
| Chat & Roleplay | **meguru** — Qwen 3.5 4B fine-tune (Ollama) | Active |
| Visual Recognition | **qwen3-vl:4b-instruct** (Ollama) | Active |
| Translation (ZH/EN/JA) | **qwen3.5:2b** (Ollama) | Active |
| Speech Synthesis | VITS (`vits-simple-api`) + Mock fallback | Active |

> **v1.0.0 note:** Chat model upgraded from InabaV1 (Qwen 2.5 7B) → **meguru** (Qwen 3.5 4B fine-tune). Visual recognition via `VisionConnector` is now live.

---

## Release Status — v1.0.0 (Stable)

Core functionality is complete and stable. The AI model, chat pipeline, voice synthesis, and desktop frontend are all fully operational.

- UI polish (window resizing, scaling) is ongoing and tracked in the roadmap below.
- Feedback and bug reports are welcome via [GitHub Issues](https://github.com/wallouo/InabaPet/issues).

### UI Roadmap

| Feature | Status |
|---|---|
| Window resize / scale support | In Progress |
| Dynamic sprite scaling | Planned |
| Settings panel | Planned |

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **[Ollama](https://ollama.com)** installed and running
- **VITS** *(optional)* — [vits-simple-api](https://github.com/Artrajz/vits-simple-api); falls back to mock audio if unavailable

### 1 — Install Models

#### Chat model — `meguru`

The model file is not bundled in the repo. Download and import manually:

```powershell
# 1. Install Ollama from https://ollama.com
# 2. Download meguru_q4_k_m.gguf + Modelfile from HuggingFace:
#    https://huggingface.co/wallouo/InabaV1/tree/main
# 3. Import into Ollama (run in the download folder):
ollama create meguru -f Modelfile
# 4. Verify
ollama list
```

#### Vision model

```powershell
ollama pull qwen3-vl:4b-instruct
```

#### Translation model

```powershell
ollama pull qwen3.5:2b
```

### 2 — Launch

This project includes a one-click startup script that handles everything automatically.

1. Right-click `run_local.ps1`
2. Select **"Run with PowerShell"**

The script will:
- Create and activate a virtual environment
- Install dependencies (`fastapi`, `uvicorn`, `requests`, `PyQt5`, `pydantic`)
- Start Ollama with the correct parallel model config
- Start the FastAPI backend on **port 5000**
- Launch the desktop pet frontend (`pet.py`)

> **Note:** If you add more Ollama models, update `$env:OLLAMA_MAX_LOADED_MODELS` in `run_local.ps1` (currently set to `3`).

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/chat_process` | POST | Main chat — auto-injects time greeting, holiday hint, user name & last topic |
| `/pat` | POST | Head-pat interaction with contextual voice response |
| `/greet` | GET | Time + holiday-aware greeting with TTS |
| `/tts` | POST | Text-to-Speech (predefined → VITS → mock fallback) |
| `/say` | POST | Generate speech from text (chains chat + TTS) |
| `/reply_bi` | POST | Bilingual (Chinese/Japanese) response generation |
| `/memory` | GET | Read memory state (`name`, `last_topic`, `mood`) |
| `/memory` | POST | Update memory fields |

---

## Dev Tools

```powershell
# Health check — verifies Ollama and API status
python healthcheck.py

# Unit tests — run from project root
python -m unittest discover -v
```

---

## Changelog (fix branch — 2026-05-02)

<details>
<summary>Bug Fixes</summary>

- **`api.py`** — Fixed broken import `from translation import translate` → `from translate import translate` (caused FastAPI startup failure on port 5000)
- **`vision/vision_connector.py`** — Removed duplicated code blocks (`__init__`, `is_qwen_vl()`, `image_to_base64()`, `analyze_image()`) introduced by a bad merge, fixing `SyntaxError` on startup
- **`vision/vision_connector.py`** — Removed unused `import cv2` and redundant `import io` inside method body

</details>

<details>
<summary>Refactored Modules</summary>

**`logic/memory.py`** — Thread-safe rewrite
- `_sanitize()` now runs outside `_file_lock` to prevent potential deadlock
- Added `last_error` property for external error inspection
- Background save via `threading.Thread(daemon=True)`

**`logic/calendar_event.py`** — Expanded holiday system
- All content converted to Japanese (めぐる speech style)
- Expanded from 3 → **12 Japanese holidays** (正月, 節分, バレンタイン, ひな祭り, ホワイトデー, エイプリルフール, こどもの日, 七夕, ハロウィン, クリスマスイブ, クリスマス, 大晦日)
- Returns `HolidayEvent` TypedDict with `name`, `hint`, and `emotion` fields
- Added `get_holiday_hint()` for direct prompt injection

**`logic/time_greeter.py`** — Full rewrite
- All greetings converted to Japanese (めぐる tone)
- Time segments expanded 4 → **7** (early morning / morning / noon / afternoon / evening / night / late night)
- Returns `TimeGreeting` TypedDict with `text` and `emotion` fields

</details>

---

## 中文說明

這是一個基於 [MurasamePet](https://github.com/LemonQu-GIT/MurasamePet) 重構的乾淨版本，使用 **PyQt5** 作為前端 GUI，結合本地 Ollama 模型實現對話、視覺識別、語音合成與桌寵互動，完全離線運行。

### 發布狀態（v1.0.0 — 正式版）

核心功能已完整穩定。AI 模型、對話流程、語音合成與桌寵前端均正常運作。視窗縮放等 UI 細節持續優化中，歡迎透過 [GitHub Issues](https://github.com/wallouo/InabaPet/issues) 回報問題。

### UI 開發路線圖

| 功能 | 狀態 |
|---|---|
| 視窗縮放支援 | 開發中 |
| 角色立繪動態縮放 | 計畫中 |
| 設定面板 | 計畫中 |

---

### 模型架構

| 角色 | 模型 | 狀態 |
|---|---|---|
| 對話與角色扮演 | **meguru**（Qwen 3.5 4B 微調版，透過 Ollama） | 運作中 |
| 視覺識別 | **qwen3-vl:4b-instruct**（透過 Ollama） | 運作中 |
| 翻譯（中 / 英 / 日） | **qwen3.5:2b**（透過 Ollama） | 運作中 |
| 語音合成 | VITS（vits-simple-api）+ Mock 保底 | 運作中 |

> **升級說明 (2026-05-02)：** 對話模型已從 InabaV1（Qwen 2.5 7B）升級至 **meguru**（Qwen 3.5 4B 微調版）。視覺識別已透過 `VisionConnector` 正式整合。

---

### 快速啟動

#### 模型安裝

```powershell
# 對話模型（需手動從 HuggingFace 下載後匯入）
# https://huggingface.co/wallouo/InabaV1/tree/main
ollama create meguru -f Modelfile

# 視覺模型
ollama pull qwen3-vl:4b-instruct

# 翻譯模型
ollama pull qwen3.5:2b
```

#### 啟動方式

1. 在 `run_local.ps1` 上點擊右鍵
2. 選擇「**使用 PowerShell 執行**」

腳本將自動建立虛擬環境、安裝套件、啟動 Ollama 與 FastAPI 後端（連接埠 5000），並開啟桌寵前端。

> **注意：** 若新增模型，請更新 `run_local.ps1` 中的 `$env:OLLAMA_MAX_LOADED_MODELS`（目前為 `3`）。

---

### API 端點

| 端點 | 方法 | 說明 |
|---|---|---|
| `/chat_process` | POST | 主對話端點，自動注入時段問候、節日提示、使用者名稱與上次話題 |
| `/pat` | POST | 摸頭互動，觸發帶語音的情境回應 |
| `/greet` | GET | 結合時段與節日的問候語（附 TTS） |
| `/tts` | POST | 語音合成（預定義 → VITS → Mock） |
| `/say` | POST | 文字轉語音（鏈結對話與 TTS） |
| `/reply_bi` | POST | 中日雙語回覆生成 |
| `/memory` | GET | 讀取記憶狀態（name、last_topic、mood） |
| `/memory` | POST | 更新記憶欄位 |

---

### 開發工具

```powershell
# 健康檢查
python healthcheck.py

# 單元測試
python -m unittest discover -v
```

---

### 本次更新（fix branch — 2026-05-02）

<details>
<summary>錯誤修復</summary>

- **`api.py`**：修復錯誤 import，解決 FastAPI 無法在連接埠 5000 啟動的問題
- **`vision/vision_connector.py`**：移除合併錯誤造成的重複程式碼，修復啟動時 `SyntaxError`
- **`vision/vision_connector.py`**：清除未使用的 `import cv2` 與冗餘 `import io`

</details>

<details>
<summary>重構模組</summary>

**`logic/memory.py`** — 執行緒安全重寫：`_sanitize()` 移至 `_file_lock` 外、新增 `last_error`、背景執行緒非同步存檔

**`logic/calendar_event.py`** — 節日系統擴充：3 → 12 個日本節日，全部改為日語，回傳 `HolidayEvent` TypedDict

**`logic/time_greeter.py`** — 完全重寫：時段 4 → 7 段，全日語問候語，回傳 `TimeGreeting` TypedDict

</details>
