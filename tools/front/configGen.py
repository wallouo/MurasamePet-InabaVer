import json

# 定義身體底圖和表情圖層
bodies = [1442, 1445, 1444, 1438, 1446, 1447, 1470, 1472, 1503, 1501, 1436, 1437, 1784, 1786, 1439]
faces = [1709, 1712, 1798, 1799, 1762, 1733, 1739, 1727, 1493, 1483, 1721, 1795]

# 生成所有組合
combinations = []
for body in bodies[:5]:  # 先測試前 5 個身體
    for face in faces[:8]:  # 先測試前 8 個表情
        combinations.append({
            "name": f"body_{body}_face_{face}",
            "layer_ids": [body, face]
        })

# 建立完整配置
config = {
    "layer_data_file": "raw_assets/めぐるa.txt",
    "image_folder": "raw_assets/layers/",
    "image_prefix": "めぐるa_0_",
    "canvas_size": [2500, 3542],
    "output_folder": "assets/meguru",
    "combinations": combinations
}

# 儲存配置檔
with open("sprite_config_batch.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"✅ 已生成 {len(combinations)} 個組合到 sprite_config_batch.json")
print(f"📊 包含 {len(bodies[:5])} 個身體 × {len(faces[:8])} 個表情")
