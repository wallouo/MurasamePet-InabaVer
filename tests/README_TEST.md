# 測試指引

本資料夾包含專案的自動化測試。使用 `unittest` 框架編寫，無需安裝額外依賴。

## 執行測試

1. 確保在專案根目錄下，並已安裝必要依賴（FastAPI、requests 等）。
2. 若使用虛擬環境，請先啟動虛擬環境。
3. 運行以下命令以執行所有測試：

   ```bash
   python -m unittest discover -v
   ```

測試將啟動 FastAPI 測試客戶端對 `/pat` 和 `/say` 端點進行調用，確認返回的 wav 檔案存在且字幕非空。即使 VOICEVOX 服務不可達，測試也能通過。