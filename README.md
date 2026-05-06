# Project Inaba (MurasamePet-Inaba-clean)

[English](#english) | [中文說明](#chinese)

---

<a name="english"></a>
## 🇬🇧 English

This is a clean, refactored version based on [MurasamePet](https://github.com/LemonQu-GIT/MurasamePet), featuring a **PyQt5** frontend GUI. It implements head-pat interactions, bilingual text generation, and API services. The project provides a complete backend API, desktop pet frontend, health check scripts, and automated tests for quick setup and testing on Windows.

### ⚠️ Development Status (Prototype)

This software is currently in the **Early Access / MVP** stage.
*   You may encounter bugs, unexpected crashes, or unused legacy files.
*   Features and API structures are subject to change without notice.
*   Feedback is welcome, but please use it with a "testing" mindset.

---

### 🤖 AI Models

| Role | Model | Status |
|---|---|---|
| Chat & Roleplay | **meguru** (Qwen 3.5 4B fine-tune, via Ollama) | ✅ Active |
| Visual Recognition | **qwen3-vl:4b-instruct** (via Ollama) | ✅ Active |
| Translation (ZH/EN/JA) | **qwen3.5 2b** (via Ollama, used by `translate.py`) | ✅ Active |
| Speech Synthesis | VITS (vits-simple-api) + Mock fallback | ✅ Active |

> **Upgrade note (2026-05-02):** The chat model has been migrated from InabaV1 (Qwen 2.5 7B) to **meguru** (Qwen 3.5 4B fine-tune). Visual recognition with `qwen3-vl:4b-instruct` is now live via the `VisionConnector` module.

---

### ✨ What's New (fix branch — 2026-05-02)

#### 🐛 Bug Fixes
- **`api.py`**: Fixed broken import `from translation import translate` → corrected to `from translate import translate` (caused FastAPI startup failure `FAILED` on port 5000)
- **`vision/vision_connector.py`**: Removed duplicate code blocks (entire `__init__`, `is_qwen_vl()`, `image_to_base64()`, `analyze_image()` params were duplicated due to a bad merge), fixing `SyntaxError: invalid syntax` on startup
- **`vision/vision_connector.py`**: Removed unused `import cv2` and redundant `import io` inside method body

#### 🔨 Refactored Modules
- **`logic/memory.py`** — Thread-safe rewrite:
  - Fixed potential deadlock: `_sanitize()` now runs outside `_file_lock`
  - Added `last_error` property for external error inspection
  - Background save via `threading.Thread(daemon=True)`

- **`logic/calendar_event.py`** — Expanded holiday system:
  - All content converted to Japanese (めぐる speech style)
  - Expanded from 3 to **12 Japanese holidays** (正月, 節分, バレンタイン, ひな祭り, ホワイトデー, エイプリルフール, こどもの日, 七夕, ハロウィン, クリスマスイブ, クリスマス, 大晦日)
  - Now returns `HolidayEvent` TypedDict with `name`, `hint`, and `emotion` fields
  - Added `get_holiday_hint()` convenience function for prompt injection

- **`logic/time_greeter.py`** — Full rewrite:
  - All greetings converted to Japanese (めぐる tone)
  - Time segments expanded from 4 → **7 segments** (early morning / morning / noon / afternoon / evening / night / late night)
  - Now returns `TimeGreeting` TypedDict with `text` and `emotion` fields

---

### Features Overview

*   **/chat_process**: Main chat endpoint. Injects time-of-day greeting, holiday hint, user name, and last topic into the Ollama prompt automatically.
*   **/pat**: Head-pat interaction, triggers a contextual response with voice.
*   **/greet**: Returns a time + holiday aware greeting with TTS.
*   **/tts**: Text-to-Speech synthesis (predefined audio → VITS → mock fallback).
*   **/say**: Generates speech from text, internally chains chat and TTS.
*   **/reply_bi**: Generates bilingual (Chinese/Japanese) responses.
*   **/memory GET**: Read current memory state (name, last_topic, mood).
*   **/memory POST**: Update memory fields.
*   **translate.py**: Translation helper module for multilingual support, used to handle Chinese / English / Japanese conversion before or after model responses.
*   **Desktop Pet Frontend**: `pet.py` uses PyQt5 to display the character, listens for mouse interactions, and plays voice/subtitles on trigger.

---

### 📥 Model Setup (Required)

Two models are required. Install both before running.

#### 1. Chat model — `meguru`

Because the model file is large, it is not included in the repo. Download and import manually:

1. **Install Ollama**: [ollama.com](https://ollama.com)
2. **Download model files** from Hugging Face:
   [https://huggingface.co/wallouo/InabaV1/tree/main](https://huggingface.co/wallouo/InabaV1/tree/main)
   — download `meguru_q4_k_m.gguf` and `Modelfile`
3. **Import to Ollama** — open PowerShell in the download folder:
   ```powershell
   ollama create meguru -f Modelfile
   ```
4. **Verify**: `ollama list` should show `meguru`

#### 2. Vision model — `qwen3-vl:4b-instruct`

```powershell
ollama pull qwen3-vl:4b-instruct
```

Verify with `ollama list`.

#### 3. Translation model — `qwen3.5 2b`

This project now includes a translation layer via `translate.py`, used for multilingual support (Chinese / English / Japanese).

```powershell
ollama pull qwen3.5:2b
```
---

### Installation & Prerequisites

1. **Python 3.9+** recommended
2. **Ollama** running with both `meguru` and `qwen3-vl:4b-instruct` loaded (see above)
3. **VITS** (optional): [vits-simple-api](https://github.com/Artrajz/vits-simple-api) — if not running, TTS falls back to mock audio automatically

---

### 🚀 How to Run

This project includes a one-click startup script that handles dependency installation and environment setup automatically.

**Steps:**
1. Right-click on `run_local.ps1`
2. Select **"Run with PowerShell"**

The script will automatically:
- Create a virtual environment
- Install required packages (`fastapi`, `uvicorn`, `requests`, `PyQt5`, `pydantic`)
- Start Ollama (restarts with correct parallel config)
- Start the FastAPI backend on port 5000
- Launch the desktop pet frontend (`pet.py`)

> **Note:** If you add a new Ollama model, update `$env:OLLAMA_MAX_LOADED_MODELS` in `run_local.ps1` accordingly (currently `3`).

---

### Health Check & Testing

For developers:

- **Health Check**: `python healthcheck.py` — verifies Ollama and API status
- **Unit Tests**: `python -m unittest discover -v` — runs API functional tests from the root directory

---

<a name="chinese"></a>
## 🇹🇼 中文說明

這是一個基於 [MurasamePet](https://github.com/LemonQu-GIT/MurasamePet) 重構的乾淨版本，使用 **PyQt5** 作為前端 GUI，實現摸頭互動、雙語生成與 API 服務。

### ⚠️ 開發中版本 (Prototype)

本程式目前處於 **早期開發階段 (MVP)**。
*   可能會遇到 Bug、未預期的崩潰或無用的殘留檔案。
*   功能與 API 結構可能隨時變動。
*   歡迎反饋問題，但請以「測試版」的心態使用。

---

### 🤖 模型架構

| 角色 | 模型 | 狀態 |
|---|---|---|
| 對話與角色扮演 | **meguru**（Qwen 3.5 4B 微調版，透過 Ollama） | ✅ 運作中 |
| 視覺識別 | **qwen3-vl:4b-instruct**（透過 Ollama） | ✅ 運作中 |
| 翻譯（中 / 英 / 日） | **qwen3.5 2b**（透過 Ollama，由 `translate.py` 使用） | ✅ 運作中 |
| 語音合成 | VITS（vits-simple-api）+ Mock 保底 | ✅ 運作中 |

> **升級說明 (2026-05-02)：** 對話模型已從 InabaV1（Qwen 2.5 7B）升級至 **meguru**（Qwen 3.5 4B 微調版）。視覺識別功能已透過 `VisionConnector` 模組正式整合 `qwen3-vl:4b-instruct`。

---

### ✨ 本次更新（fix branch — 2026-05-02）

#### 🐛 錯誤修復
- **`api.py`**：修復 `from translation import translate` 的錯誤 import（正確應為 `from translate import translate`），此問題導致 FastAPI 無法啟動（連接埠 5000 顯示 `FAILED`）
- **`vision/vision_connector.py`**：移除因合併錯誤造成的大量重複程式碼（`__init__`、`is_qwen_vl()`、`image_to_base64()`、`analyze_image()` 參數均被重複貼上），修復啟動時的 `SyntaxError: invalid syntax`
- **`vision/vision_connector.py`**：清除未使用的 `import cv2` 及方法內部的冗餘 `import io`

#### 🔨 重構模組

- **`logic/memory.py`** — 執行緒安全重寫：
  - 修復潛在 deadlock：`_sanitize()` 現在在 `_file_lock` 外執行
  - 新增 `last_error` property，方便外部檢查錯誤狀態
  - 以背景執行緒（`daemon=True`）非同步存檔

- **`logic/calendar_event.py`** — 節日系統擴充：
  - 全部改為日語（めぐる口吻）
  - 節日數量從 3 個擴充至 **12 個日本節日**（正月、節分、バレンタイン、ひな祭り、ホワイトデー、エイプリルフール、こどもの日、七夕、ハロウィン、クリスマスイブ、クリスマス、大晦日）
  - 回傳 `HolidayEvent` TypedDict，包含 `name`、`hint`、`emotion` 三個欄位
  - 新增 `get_holiday_hint()` 便利函式，可直接注入 prompt

- **`logic/time_greeter.py`** — 完全重寫：
  - 問候語全部改為日語（めぐる語氣）
  - 時段從 4 段擴充至 **7 段**（清晨 / 上午 / 午間 / 下午 / 傍晚 / 夜間 / 深夜）
  - 回傳 `TimeGreeting` TypedDict，包含 `text` 和 `emotion` 欄位

---

### 功能概述

*   **/chat_process**：主要對話端點，會自動將時段問候、節日提示、使用者名稱及上次話題注入 Ollama prompt。
*   **/pat**：摸頭互動，觸發帶有語音的情境回應。
*   **/greet**：回傳結合時段與節日的問候語，附帶 TTS。
*   **/tts**：語音合成服務（預定義音頻 → VITS → Mock 保底）。
*   **/say**：根據文字生成語音並返回字幕。
*   **/reply_bi**：生成中日雙語回覆。
*   **/memory GET**：讀取目前記憶狀態（name、last_topic、mood）。
*   **/memory POST**：更新記憶欄位。
*   **translate.py**：多語翻譯輔助模組，負責處理中文 / 英文 / 日文的轉換與支援。
*   **前端桌寵**：`pet.py` 使用 PyQt5 顯示角色立繪，監聽滑鼠操作並播放語音與字幕。

---

### 📥 模型設置（必要步驟）

需要安裝兩個模型，啟動前請先完成。

#### 1. 對話模型 — `meguru`

由於模型檔案較大，未包含在 repo 中，請手動下載並匯入：

1. **安裝 Ollama**：[ollama.com](https://ollama.com)
2. **從 Hugging Face 下載模型檔案**：
   [https://huggingface.co/wallouo/InabaV1/tree/main](https://huggingface.co/wallouo/InabaV1/tree/main)
   — 下載 `meguru_q4_k_m.gguf` 和 `Modelfile`
3. **匯入 Ollama** — 在下載資料夾開啟 PowerShell：
   ```powershell
   ollama create meguru -f Modelfile
   ```
4. **驗證**：執行 `ollama list`，確認列表中有 `meguru`

#### 2. 視覺模型 — `qwen3-vl:4b-instruct`

```powershell
ollama pull qwen3-vl:4b-instruct
```

#### 3. 翻譯模型 — `qwen3.5 2b`

本專案現在新增 `translate.py` 翻譯模組，用於支援中文 / 英文 / 日文處理。

```powershell
ollama pull qwen3.5:2b
```
執行 `ollama list` 確認已下載完成。

---

### 安裝與準備

1. 建議使用 **Python 3.9+**
2. **Ollama** 需已啟動，且 `meguru` 與 `qwen3-vl:4b-instruct` 兩個模型均已安裝（見上方步驟）
3. **VITS**（可選）：[vits-simple-api](https://github.com/Artrajz/vits-simple-api)——若未啟動，TTS 會自動退回 Mock 音頻

---

### 🚀 啟動方式

本專案提供一鍵啟動腳本，自動處理依賴安裝與環境設置。

**步驟：**
1. 在 `run_local.ps1` 上點擊右鍵
2. 選擇「**使用 PowerShell 執行**」

腳本將自動：
- 建立虛擬環境
- 安裝必要套件（`fastapi`、`uvicorn`、`requests`、`PyQt5`、`pydantic`）
- 啟動 Ollama（以正確的並行設定重啟）
- 在連接埠 5000 啟動 FastAPI 後端
- 啟動前端桌寵程式（`pet.py`）

> **注意：** 若新增 Ollama 模型，請同步更新 `run_local.ps1` 中的 `$env:OLLAMA_MAX_LOADED_MODELS`（目前設為 `3`）。

---

### 健康檢查與測試

開發者可使用以下腳本進行除錯：

- **健康檢查**：`python healthcheck.py` — 檢查 Ollama 與 API 服務狀態
- **單元測試**：在根目錄執行 `python -m unittest discover -v`
