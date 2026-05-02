#!/bin/bash

# ==========================================
# AutoDL 終極訓練與轉換腳本 (RTX 5090 強化版)
# ==========================================

# 開啟「錯誤即停止」模式：只要任何一行指令失敗，腳本就會中斷，避免浪費時間與金錢
set -e

# 定義絕對路徑變數，避免腳本執行時因為相對路徑 (cd ..) 迷路
WORKSPACE="/root/LLaMA-Factory"
LLAMA_CPP_DIR="/root/llama.cpp"
BACKUP_DIR="/root/autodl-tmp/meguru_final"

# ==========================================
# 1. 環境設定與錯誤處理
# ==========================================

# 定義錯誤處理函數，當腳本報錯中斷時，也能自動關機止損
error_handler() {
    echo "=========================================="
    echo "❌ 發生錯誤！腳本已中斷。"
    echo "為了節省 AutoDL 費用，系統將在 1 分鐘後自動關機..."
    echo "=========================================="
    # 這裡預設開啟關機止損。如果你還在除錯階段，可以把下面這行註解掉
    shutdown -h +1
}
# 捕捉 ERR 訊號並觸發 error_handler
trap 'error_handler' ERR

# 喚醒自訂的 Conda 環境 (依照你之前的 SOP，使用資料碟的環境)
echo "🔄 正在啟動 Conda 環境..."
source /root/miniconda3/bin/activate /root/autodl-tmp/myenv

# 開啟 AutoDL 學術加速（下載 github 與 huggingface 必備）
source /etc/network_turbo 2>/dev/null || true

# 進入工作目錄
cd $WORKSPACE

# ==========================================
# 2. 開始訓練 (4 連抽)
# ==========================================
echo "🚀 [1/6] 開始執行 4 連抽 LoRA 訓練..."
echo "------------------------------------------"
DISABLE_VERSION_CHECK=1 llamafactory-cli train config_v1_baseline_bf16.yaml
echo "✅ V1 訓練完成！"

DISABLE_VERSION_CHECK=1 llamafactory-cli train config_v2_mlp_persona_bf16.yaml
echo "✅ V2 訓練完成！"

DISABLE_VERSION_CHECK=1 llamafactory-cli train config_v3_quant_safe_bf16.yaml
echo "✅ V3 訓練完成！"

DISABLE_VERSION_CHECK=1 llamafactory-cli train config_v4_anti_loop_bf16.yaml
echo "✅ V4 訓練完成！"


# ==========================================
# 3. 合併模型 (動態生成 merge.yaml)
# ==========================================
echo "🚀 [2/6] 開始合併最佳模型 (預設合併 V1)..."
echo "------------------------------------------"

# 動態生成 merge.yaml，確保路徑與格式絕對正確
cat <<EOF > merge.yaml
model_name_or_path: Qwen/Qwen3.5-4B
adapter_name_or_path: saves/qwen3.5-4b/lora/v1_baseline_bf16
template: qwen
finetuning_type: lora
export_dir: models/meguru-4b-merged
export_size: 2
export_device: cpu
export_legacy_format: false
EOF

DISABLE_VERSION_CHECK=1 llamafactory-cli export merge.yaml
echo "✅ 模型合併完成！"


# ==========================================
# 4. llama.cpp 環境準備
# ==========================================
echo "🚀 [3/6] 準備 llama.cpp 環境..."
echo "------------------------------------------"
cd /root 
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd $LLAMA_CPP_DIR

# 安裝 gguf 相依套件
pip install -r requirements.txt || pip install gguf

# 編譯轉換與量化工具
make -j
echo "✅ llama.cpp 編譯完成！"


# ==========================================
# 5. 模型轉換與量化
# ==========================================
echo "🚀 [4/6] 轉換模型為 FP16 GGUF 格式..."
echo "------------------------------------------"
# 使用絕對路徑指向 merge 出來的資料夾
python convert_hf_to_gguf.py $WORKSPACE/models/meguru-4b-merged --outtype f16 --outfile meguru-4b-fp16.gguf
echo "✅ GGUF FP16 轉換完成！"

echo "🚀 [5/6] 進行 Q4_K_M 量化..."
echo "------------------------------------------"
./llama-quantize meguru-4b-fp16.gguf meguru-4b-q4_k_m.gguf Q4_K_M
echo "✅ 量化完成！您的最終檔案位於: $(pwd)/meguru-4b-q4_k_m.gguf"


# ==========================================
# 6. 備份與關機
# ==========================================
echo "🚀 [6/6] 任務全部圓滿達成！準備關機以節省費用..."
echo "------------------------------------------"

# 將重要檔案移動到資料碟 (autodl-tmp) 確保重開機後還在
mkdir -p $BACKUP_DIR
cp meguru-4b-q4_k_m.gguf $BACKUP_DIR/
cp -r $WORKSPACE/saves/qwen3.5-4b/lora $BACKUP_DIR/ 2>/dev/null || true

echo "✅ 檔案已備份至 $BACKUP_DIR"
echo "系統將在 10 秒後自動關機 (Shutdown)..."

# 關閉學術加速
unset http_proxy https_proxy all_proxy

sleep 10
shutdown -h now