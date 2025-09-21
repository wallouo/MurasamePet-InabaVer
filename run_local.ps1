# 啟動腳本：創建虛擬環境、安裝依賴並啟動 API
param()

Write-Host "[Inaba-Clean] 建立虛擬環境並安裝依賴..."
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install fastapi uvicorn requests PyQt5

$env:PYTHONPATH = $PSScriptRoot
$apiPort = $Env:API_PORT
if (-not $apiPort) { $apiPort = 5000 }
Write-Host "[Inaba-Clean] 啟動 API 於 port $apiPort ..."
Start-Process -NoNewWindow -FilePath python -ArgumentList "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", $apiPort
Write-Host "[Inaba-Clean] 若需要圖形界面，請另行運行 python pet.py"