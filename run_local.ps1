# 因幡巡桌寵啟動腳本 - 完整版（容錯加強 + Ollama 預熱）
param()

# ===== Ollama Parallelism Config =====
$env:OLLAMA_NUM_PARALLEL = 3
$env:OLLAMA_MAX_LOADED_MODELS = 3
$env:OLLAMA_KEEP_ALIVE = "5m"

# ===== 配置區 =====
$VITS_PATH = "D:\vits-simple-api-windows-gpu-v0.6.16"
$VITS_STARTUP = "start.bat"
$VITS_PORT = 23456
$API_PORT = if ($Env:API_PORT) { $Env:API_PORT } else { 5000 }
$OLLAMA_PORT = 11434
$OLLAMA_MODEL = "meguru"  # ← 新增：指定模型名稱

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Inaba Meguru Desktop Pet" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ===== 函數：偵測連接埠 =====
function Test-Port {
    param([int]$Port)
    try {
        $conn = New-Object System.Net.Sockets.TcpClient
        $conn.Connect("127.0.0.1", $Port)
        $conn.Close()
        return $true
    } catch {
        return $false
    }
}

# ===== 步驟 0：虛擬環境管理 =====
Write-Host "[0/5] Setting up virtual environment..." -ForegroundColor Yellow

if (-Not (Test-Path ".venv")) {
    Write-Host "  Creating virtual environment..." -NoNewline
    python -m venv .venv
    Write-Host " OK" -ForegroundColor Green
}

Write-Host "  Activating virtual environment..." -NoNewline
& .\.venv\Scripts\Activate.ps1
Write-Host " OK" -ForegroundColor Green

Write-Host "  Checking dependencies..." -NoNewline

# 嘗試升級 pip，失敗則跳過
$ErrorActionPreference = "SilentlyContinue"
pip install --upgrade pip -q 2>$null
$ErrorActionPreference = "Stop"

# 安裝依賴
pip install --no-warn-script-location fastapi uvicorn requests PyQt5 pydantic -q

if ($LASTEXITCODE -eq 0) {
    Write-Host " OK" -ForegroundColor Green
} else {
    Write-Host " WARNING (some packages may already be installed)" -ForegroundColor Yellow
}

$env:PYTHONPATH = $PSScriptRoot

# ===== 步驟 1：啟動 VITS =====
Write-Host "[1/5] Starting VITS service..." -ForegroundColor Yellow

if (Test-Port $VITS_PORT) {
    Write-Host "  VITS already running (port $VITS_PORT)" -ForegroundColor Green
} else {
    $vitsFullPath = Join-Path $VITS_PATH $VITS_STARTUP
    if (-Not (Test-Path $vitsFullPath)) {
        Write-Host "  WARNING: VITS startup file not found: $vitsFullPath" -ForegroundColor Yellow
        Write-Host "  Skipping VITS (will use Mock TTS)..." -ForegroundColor Yellow
    } else {
        Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "`"$vitsFullPath`"" -WindowStyle Normal -WorkingDirectory $VITS_PATH

        Write-Host "  Waiting for VITS to start..." -NoNewline
        $timeout = 30
        $elapsed = 0
        while (-not (Test-Port $VITS_PORT) -and $elapsed -lt $timeout) {
            Start-Sleep -Seconds 1
            Write-Host "." -NoNewline
            $elapsed++
        }

        if (Test-Port $VITS_PORT) {
            Write-Host " OK" -ForegroundColor Green
        } else {
            Write-Host " TIMEOUT (will use Mock TTS)" -ForegroundColor Yellow
        }
    }
}

# ===== 步驟 1.5：配置 Ollama =====
Write-Host "[1.5/5] Configuring Ollama..." -ForegroundColor Yellow

# Check if Ollama is running
if (Test-Port $OLLAMA_PORT) {
    Write-Host "  Ollama is running. Restarting to apply parallel config..." -ForegroundColor Yellow
    Stop-Process -Name "ollama_app_v*" -ErrorAction SilentlyContinue
    Stop-Process -Name "ollama" -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Start Ollama with new env vars
Write-Host "  Starting Ollama with NUM_PARALLEL=2..." -ForegroundColor Green
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden

# Wait for startup
$timeout = 10
$elapsed = 0
while (-not (Test-Port $OLLAMA_PORT) -and $elapsed -lt $timeout) {
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline
    $elapsed++
}
Write-Host " OK" -ForegroundColor Green

# ===== 步驟 2：啟動後端 API =====
Write-Host "[2/5] Starting Backend API..." -ForegroundColor Yellow

if (Test-Port $API_PORT) {
    Write-Host "  Backend already running (port $API_PORT)" -ForegroundColor Yellow
} else {
    Write-Host "  Starting FastAPI on port $API_PORT..." -NoNewline
    Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "$API_PORT" -WindowStyle Normal
    Start-Sleep -Seconds 3

    if (Test-Port $API_PORT) {
        Write-Host " OK" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# ===== 步驟 3：健康檢查 =====
Write-Host "[3/5] Health check..." -ForegroundColor Yellow

if (Test-Path "healthcheck.py") {
    & python healthcheck.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "WARNING: Health check failed!" -ForegroundColor Red
        Write-Host "  Common causes:" -ForegroundColor Yellow
        Write-Host "  1. Ollama not running" -ForegroundColor Yellow
        Write-Host "  2. Port occupied" -ForegroundColor Yellow
        Read-Host "Press Enter to continue anyway (or Ctrl+C to exit)"
    }
} else {
    Write-Host "  healthcheck.py not found, skipping..." -ForegroundColor Yellow
}

# ===== 步驟 4：啟動前端 =====
Write-Host "[4/5] Starting Frontend..." -ForegroundColor Yellow
Write-Host ""
Write-Host "All services started! Inaba will appear shortly..." -ForegroundColor Green
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Tips:" -ForegroundColor Cyan
Write-Host "  - Close pet window = frontend only" -ForegroundColor White
Write-Host "  - Backend keeps running" -ForegroundColor White
Write-Host "  - Run stop.bat to stop all" -ForegroundColor White
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

& python pet.py

Write-Host ""
Write-Host "Frontend closed." -ForegroundColor Yellow
Write-Host "Backend still running. You can run 'python pet.py' again." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
