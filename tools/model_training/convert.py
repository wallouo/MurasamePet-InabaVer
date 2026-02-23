from unsloth import FastLanguageModel
import torch

trained_model_path = "outputs/checkpoint-819" 

print("🌸 正在載入模型與 LoRA 權重...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = trained_model_path, 
    max_seq_length = 4096,     
    dtype = torch.float16,     
    load_in_4bit = True,   
)

print("🚀 開始將 LoRA 與基礎模型合併為標準 HF 格式...")

# 🔴 關鍵改變：不要轉 GGUF 了！我們存成標準的 merged_16bit 格式
model.save_pretrained_merged("meguru_merged_hf", tokenizer, save_method = "merged_16bit")

print("🎉 合併完成！完整的模型已儲存在 meguru_merged_hf 資料夾中。")