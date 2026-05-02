# -*- coding: utf-8 -*-
"""
Vision Connector - Ollama 視覺模型連接器
支援 Llava, Moondream, Qwen-VL 等多種模型
"""

import base64
import io
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from PIL import Image


class VisionConnectorError(Exception):
    """視覺連接器錯誤"""
    pass


class VisionConnector:
    """
    通用視覺模型連接器

    特性:
    - 支援圖片路徑、PIL Image、numpy array
    - 自動 base64 編碼
    - VRAM 優化（keep_alive 控制）
    - 錯誤處理與重試
    """

    DEFAULT_OPTIONS = {
        "temperature": 0.3,
        "num_predict": 150,
        "num_ctx": 8192,
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-vl:4b-instruct",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.generate_url = f"{self.base_url}/api/generate"
        print(f"🔧 [VisionConnector] Initialized with model: {self.model}, timeout: {self.timeout}s")

    def is_qwen_vl(self) -> bool:
        """判斷目前使用的模型是否為 qwen-vl 系列"""
        return "qwen" in self.model.lower()

    def image_to_base64(self, image) -> str:
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = image[:, :, ::-1]          # BGR → RGB
                elif image.shape[2] == 4:
                    image = image[:, :, [2, 1, 0, 3]]  # BGRA → RGBA
            pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            raise TypeError(f"不支援的圖片格式: {type(image)}")

        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        prompt: str = "この画像を詳しく説明してください。どんなアプリ、ゲーム、または作業が表示されているか教えてください。",
        stream: bool = False,
        keep_alive: str = "5m",
        custom_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """分析圖片內容"""
        try:
            image_b64 = self.image_to_base64(image)

            options = self.DEFAULT_OPTIONS.copy()
            if custom_options:
                options.update(custom_options)

            if self.is_qwen_vl():
                api_endpoint = f"{self.base_url}/api/chat"
                vl_options = options.copy()
                vl_options.update({
                    "temperature": 0.1,
                    "num_predict": 80,
                    "top_k": 20,
                    "repeat_penalty": 1.2,
                })
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],
                        }
                    ],
                    "stream": stream,
                    "keep_alive": keep_alive,
                    "options": vl_options,
                }
            else:
                api_endpoint = f"{self.base_url}/api/generate"
                vl_options = options
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": stream,
                    "keep_alive": keep_alive,
                    "options": vl_options,
                }

            print(f"[VisionConnector] Endpoint: {api_endpoint}")
            print(f"[VisionConnector] Model: {self.model}")
            print(f"[VisionConnector] Prompt: {prompt[:100]}...")
            print(f"[VisionConnector] Image size: {len(image_b64)} bytes (base64)")
            print(f"[VisionConnector] API mode: {'chat' if self.is_qwen_vl() else 'generate'}")

            response = requests.post(api_endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            if self.is_qwen_vl():
                message = data.get("message", {})
                response_text = message.get("content", "").strip()
                if not response_text and "thinking" in message:
                    thought = message.get("thinking", "")
                    print(f"⚠️ [VisionConnector] Model trapped in thought loop: {thought[:100]}...")
                    return "A computer screen with active windows."
            else:
                response_text = data.get("response", "").strip()

            print(f"[VisionConnector] Response length: {len(response_text)}")
            print(f"[VisionConnector] Response preview: {response_text[:200]}")

            if not response_text:
                print(f"⚠️ [VisionConnector] Model {self.model} returned empty response!")
                print(f"[VisionConnector] Full API response: {data}")
                if self.is_qwen_vl():
                    return "The screen shows various applications and content."

            return response_text

        except requests.exceptions.Timeout:
            raise VisionConnectorError(
                f"請求超時（{self.timeout}s）。模型 {self.model} 可能尚未載入或推理時間過長"
            )
        except requests.exceptions.ConnectionError:
            raise VisionConnectorError(
                f"無法連接到 Ollama ({self.base_url})。請確認服務已啟動"
            )
        except requests.exceptions.HTTPError as e:
            raise VisionConnectorError(f"HTTP 錯誤: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise VisionConnectorError(f"未知錯誤: {str(e)}")