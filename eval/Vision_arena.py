import ollama
import os
import json

# ==========================================
# 1. 競技場設定
# ==========================================
MODEL_A = 'qwen3-vl:4b'
MODEL_B = 'minicpm-v'
IMAGE_DIR = '.'
OUTPUT_FILE = 'vision_arena_results.json'

# 預設問題 (如果圖片沒有被設定專屬問題，就會用這個)
DEFAULT_PROMPT = "請詳細描述這張圖片的內容，包含任何文字、物體以及它們的空間相對位置。"

# ==========================================
# 2. 陷阱題題庫 (Killer Prompts Mapping)
# ==========================================
# 💡 請確保這裡的 key (例如 'image_94f4ff.jpg') 與你資料夾內的檔名完全一致！
PROMPTS_MAPPING = {
    # 1. YouTube 首頁
    "1.png": 
        "請找出畫面中標題包含『印度高分神作』的影片，並告訴我它的觀看次數和上傳時間分別是多少？",
        
    # 2. 英雄聯盟戰績表 (極限表格對齊)
    "2.png": 
        "請告訴我玩家 'actor8' (使用好運姐) 的 KDA (擊殺 / 死亡 / 助攻) 以及他的吃兵數 (CS，也就是金錢旁邊的數字) 是多少？",
        
    # 3. GitHub Commit (跨行文字比對)
    "3.png": 
        "請精準列出，哪兩個檔案的 commit 訊息是 'feat: Fix subtitle position and Modelfile parameters synchron...'？",
        
    # 4. 終端機日誌 (密集文字與版本號)
    "4.png": 
        "請問終端機正在下載安裝的 `torch` 版本號碼和 CUDA 版本具體是什麼？(請從畫面中的 Downloading 網址或文字中提取)",
        
    # 5. 和服少女 (幻覺測試)
    "5.jpg": 
        "請詳細描述畫面左側的建築物特徵以及水池中的物品。請誠實回答，畫面中有出現任何動物嗎？",
        
    # 6. 魔法少女遊戲主選單 (風格化藝術字與邊緣細節)
    "6.jpg": 
        "請提取畫面中最大的中文標題，並告訴我畫面右下角的遊戲版本號是多少？"
}

# ==========================================
# 3. 執行競技場主程式
# ==========================================
def run_vision_arena():
    results = []
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"🚨 沒有在 {IMAGE_DIR} 資料夾中找到圖片！")
        return

    print(f"🔥 競技場啟動！共找到 {len(image_files)} 張測試圖片。")

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        # 自動抓取這張圖片專屬的陷阱題！
        current_prompt = PROMPTS_MAPPING.get(img_name, DEFAULT_PROMPT)
        
        print(f"\n==============================================")
        print(f"📸 正在處理圖片: {img_name}")
        print(f"❓ 測試問題: {current_prompt}")
        print(f"==============================================")
        
        image_result = {
            "image_name": img_name,
            "prompt": current_prompt,
            MODEL_A: "",
            MODEL_B: ""
        }

        # --- 呼叫 Model A ---
        print(f"🤖 [{MODEL_A}] 思考中...")
        try:
            response_a = ollama.chat(
                model=MODEL_A,
                messages=[{
                    'role': 'user',
                    'content': current_prompt,
                    'images': [img_path]
                }]
            )
            image_result[MODEL_A] = response_a['message']['content']
            print(f"✅ {MODEL_A} 回答完畢！")
        except Exception as e:
            print(f"❌ {MODEL_A} 發生錯誤: {e}")
            image_result[MODEL_A] = f"ERROR: {str(e)}"

        # --- 呼叫 Model B ---
        print(f"🤖 [{MODEL_B}] 思考中...")
        try:
            response_b = ollama.chat(
                model=MODEL_B,
                messages=[{
                    'role': 'user',
                    'content': current_prompt,
                    'images': [img_path]
                }]
            )
            image_result[MODEL_B] = response_b['message']['content']
            print(f"✅ {MODEL_B} 回答完畢！")
        except Exception as e:
            print(f"❌ {MODEL_B} 發生錯誤: {e}")
            image_result[MODEL_B] = f"ERROR: {str(e)}"

        results.append(image_result)

    # 存檔
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 測試完成！請打開 '{OUTPUT_FILE}' 查看兩位選手的作答狀況！")

if __name__ == "__main__":
    run_vision_arena()