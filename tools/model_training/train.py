from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 配置參數
max_seq_length = 2048
dtype = None # 自動偵測 (Float16 或 Bfloat16)
load_in_4bit = True # 4bit 量化加載，省顯存關鍵

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct", 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. 設置 LoRA 適配器 (讓模型可以學習新知識)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: 數值越大模型越聰明但顯存吃越多 (8, 16, 32, 64)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# 4. 準備數據格式化函數 (把你的 JSONL 轉成 Qwen 看得懂的 Prompt)
# Qwen 2.5 Instruct 使用 ChatML 格式: <|im_start|>system...<|im_end|>
alpaca_prompt = """<|im_start|>system
你要扮演遊戲《魔女的夜宴》中的角色「因幡めぐる」。請模仿她的語氣、口癖和性格進行回答。<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

EOS_TOKEN = tokenizer.eos_token # 必須添加結束符

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input, output in zip(inputs, outputs):
        # 如果 input 是空的（沒有前文），稍微改一下提示
        if not input:
            text = alpaca_prompt.format("（沒有前文，請說一句話）", output) + EOS_TOKEN
        else:
            text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 5. 加載你的數據集
dataset = load_dataset("json", data_files="meguru_training_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 6. 設置訓練器

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,  # 總批次大小 = 8
        gradient_checkpointing = True,    # 必開！省顯存關鍵
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit",       # 開啟分頁功能，防止 OOM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# 7. 開始訓練
print("開始訓練...")
trainer_stats = trainer.train()

# 8. 保存並轉換為 GGUF (給 Ollama 用)
print("訓練完成，正在轉換為 GGUF 格式...")
model.save_pretrained_gguf("meguru_model", tokenizer, quantization_method = "q4_k_m")
print("GGUF 模型已保存到 meguru_model 資料夾！")
