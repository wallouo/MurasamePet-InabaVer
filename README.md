<<<<<<< HEAD
# MurasamePet-InabaVer
=======
# MurasamePet-Inaba-clean

這是一個基於 [MurasamePet](https://github.com/LemonQu-GIT/MurasamePet) 重構的乾淨版本，使用 **PyQt5** 作為前端 GUI，實現摸頭互動、雙語生成、語音合成與 API 服務。專案提供完整的後端 API、前端桌寵、健康檢查腳本和自動化測試，方便在 Windows 本機快速搭建和測試。

## 功能概述

* **/qwen3**：代理本地 Ollama 的聊天接口，返回聊天回應和歷史。
* **/reply_bi**：生成中日雙語回覆（暫時以輸入為准）。
* **/tts**：語音合成。若 VOICEVOX 可用則調用 VoiceVox Engine；否則自動使用 mock TTS 產生 440 Hz 正弦波 wav。
* **/say**：根據文字生成語音並返回字幕，內部串接 `/qwen3` 和 `/tts`。
* **/pat**：模擬摸頭互動，調用 `/say` 產生一句短句並返回語音與字幕。
* **前端桌寵**：`pet.py` 使用 PyQt5 顯示角色立繪，監聽頭部區域滑鼠操作或雙擊以觸發 `/pat`，播放語音並顯示字幕。

## 安裝與準備

1. 建議使用 **Python 3.9.x**。在專案根目錄建立虛擬環境並啟用：

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install fastapi uvicorn requests PyQt5
   ```

2. 複製 `.env.example` 為 `.env`，可按需修改：

   ```powershell
   copy .env.example .env
   ```

3. （可選）安裝 [Ollama](https://ollama.com) 並執行 `ollama pull qwen3:8b` 或其他模型；安裝 [VOICEVOX Engine](https://voicevox.hiroshiba.jp/) 並啟動。若 VOICEVOX 未啟動也能運作，系統會使用 mock TTS。

## 啟動方式

開兩個終端窗口：

1. 啟動後端 API：

   ```powershell
   python -m uvicorn api:app --host 0.0.0.0 --port 5000
   ```

2. 啟動桌寵前端：

   ```powershell
   python pet.py
   ```

## 健康檢查

運行 `scripts/healthcheck.py` 以檢查 Ollama 與 VOICEVOX 的可用性：

```powershell
python scripts/healthcheck.py
```

## API 冒煙測試

可使用 `curl` 測試後端端點，例如：

```powershell
curl -X POST http://127.0.0.1:5000/pat

curl -X POST http://127.0.0.1:5000/say -H "Content-Type: application/json" -d '{"text": "我又在遊戲裡失手了…"}'
```

這些調用將返回語音檔案路徑和中文字幕，並在 `/pat` 觸發時由前端播放。

## 執行自動化測試

安裝依賴後，在根目錄執行：

```powershell
python -m unittest discover -v
```

測試將啟動 FastAPI 測試客戶端對 `/pat` 和 `/say` 進行調用，確認返回的 `wav_path` 存在且 `subtitle_zh` 非空。

## 一鍵啟動腳本（Windows）

`run_local.ps1` 提供在 Windows 下一鍵安裝依賴、啟動 API 的腳本，可按以下方式使用：

```powershell
powershell -ExecutionPolicy Bypass -File run_local.ps1
```

該腳本將創建虛擬環境、安裝依賴並啟動 API。如果需要前端桌寵，請另行運行 `python pet.py`。
>>>>>>> 8e0cbbd (Initial commit)
