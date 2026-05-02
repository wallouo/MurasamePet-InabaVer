"""
Unit tests for VisionConnector
Tests: payload format, image encoding, API routing, error handling
Run: python -m pytest vision/test_vision_connector.py -v
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import base64
from PIL import Image
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.vision_connector import VisionConnector, VisionConnectorError


def make_test_image_pil(width=64, height=64) -> Image.Image:
    """建立一個有顏色的測試用 PIL Image（不是純黑）"""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[10:30, 10:30] = [255, 0, 0]   # 紅色方塊
    arr[30:50, 30:50] = [0, 255, 0]   # 綠色方塊
    return Image.fromarray(arr)


def make_test_image_numpy() -> np.ndarray:
    """建立一個測試用 BGR numpy array（模擬 mss/OpenCV 截圖）"""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:30, 10:30] = [0, 0, 255]   # BGR: 藍色方塊
    return arr


def make_test_image_rgba() -> np.ndarray:
    """建立 RGBA numpy array（模擬 mss BGRA 截圖）"""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # alpha channel
    arr[10:30, 10:30, 2] = 255  # 紅色 (BGRA)
    return arr


class TestVisionConnectorInit(unittest.TestCase):
    """測試初始化與設定"""

    def test_default_model_name(self):
        """確認預設模型名稱格式正確（冒號而非橫槓）"""
        vc = VisionConnector()
        self.assertIn(":", vc.model, "模型名稱應使用冒號格式，例如 qwen3-vl:4b-instruct")
        self.assertNotEqual(vc.model, "qwen3-vl-4b", "舊的橫槓格式會導致 404 錯誤")

    def test_is_qwen_vl_detection(self):
        """確認 qwen-vl 模型識別正確"""
        vc_qwen = VisionConnector(model="qwen3-vl:4b-instruct")
        vc_llava = VisionConnector(model="llava:7b")
        vc_moon = VisionConnector(model="moondream:latest")

        self.assertTrue(vc_qwen.is_qwen_vl(), "qwen3-vl:4b-instruct 應被識別為 qwen_vl")
        self.assertFalse(vc_llava.is_qwen_vl(), "llava 不應被識別為 qwen_vl")
        self.assertFalse(vc_moon.is_qwen_vl(), "moondream 不應被識別為 qwen_vl")

    def test_api_endpoint_routing(self):
        """確認 qwen-vl 使用 /api/chat，其他模型使用 /api/generate"""
        vc_qwen = VisionConnector(model="qwen3-vl:4b-instruct")
        vc_llava = VisionConnector(model="llava:7b")

        self.assertIn("/api/chat", vc_qwen.generate_url if hasattr(vc_qwen, "chat_url") else "http://localhost:11434/api/chat")
        # chat_url 應存在於 qwen 分支
        self.assertTrue(vc_qwen.is_qwen_vl())
        self.assertFalse(vc_llava.is_qwen_vl())


class TestImageToBase64(unittest.TestCase):
    """測試圖片轉 base64 編碼"""

    def setUp(self):
        self.vc = VisionConnector(model="qwen3-vl:4b-instruct")

    def test_pil_image_encoding(self):
        """PIL Image 應成功轉為非空的 base64 字串"""
        img = make_test_image_pil()
        result = self.vc.image_to_base64(img)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100, "base64 字串不應幾乎為空")
        # 確認是有效的 base64
        decoded = base64.b64decode(result)
        self.assertGreater(len(decoded), 0)

    def test_numpy_bgr_encoding(self):
        """BGR numpy array 應成功轉換（模擬 OpenCV/mss 輸出）"""
        arr = make_test_image_numpy()
        result = self.vc.image_to_base64(arr)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_numpy_rgba_encoding(self):
        """RGBA numpy array（mss 截圖格式）應成功轉換"""
        arr = make_test_image_rgba()
        result = self.vc.image_to_base64(arr)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_image_is_not_blank(self):
        """確認轉換後的圖片不是全黑空白（代表圖片內容有被保留）"""
        img = make_test_image_pil()
        result = self.vc.image_to_base64(img)
        decoded = base64.b64decode(result)
        pil_back = Image.open(io.BytesIO(decoded))
        arr = np.array(pil_back)
        # 應有非零像素
        self.assertGreater(arr.max(), 0, "解碼後的圖片不應是全黑（圖片內容遺失）")


class TestAnalyzeImagePayload(unittest.TestCase):
    """
    ⭐ 核心測試：確認圖片有正確地傳送到 API
    這些測試會攔截 requests.post 來檢查實際送出的 payload
    """

    def setUp(self):
        self.vc = VisionConnector(model="qwen3-vl:4b-instruct")
        self.test_image = make_test_image_pil()

    def _make_mock_chat_response(self, content="A red square on black background."):
        """建立模擬的 /api/chat 成功回應"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "role": "assistant",
                "content": content
            }
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("vision.vision_connector.requests.post")
    def test_image_is_included_in_payload(self, mock_post):
        """
        ✅ 最重要的測試：確認 images base64 有出現在送出的 payload 中
        如果這個測試失敗，代表圖片根本沒有傳給模型
        """
        mock_post.return_value = self._make_mock_chat_response()

        self.vc.analyze_image(self.test_image)

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1].get("json") or call_kwargs[0][1]

        # 確認 messages 存在
        self.assertIn("messages", payload, "payload 中缺少 'messages' 欄位")

        messages = payload["messages"]
        self.assertGreater(len(messages), 0, "messages 陣列不應為空")

        user_msg = messages[0]
        # ⭐ 關鍵：images 必須在 message 物件的頂層
        self.assertIn(
            "images", user_msg,
            "❌ images 沒有放在 message 物件的頂層！\n"
            "正確格式: messages[0]['images'] = [base64_str]\n"
            "錯誤格式: messages[0]['content'] 是字串而 images 在別處"
        )

        images_list = user_msg["images"]
        self.assertIsInstance(images_list, list, "images 應為 list 格式")
        self.assertEqual(len(images_list), 1, "應有且只有一張圖片")
        self.assertIsInstance(images_list[0], str, "images[0] 應為 base64 字串")
        self.assertGreater(len(images_list[0]), 100, "base64 圖片字串過短，可能是空圖")

    @patch("vision.vision_connector.requests.post")
    def test_payload_uses_chat_endpoint(self, mock_post):
        """確認 qwen-vl 使用 /api/chat 而非 /api/generate"""
        mock_post.return_value = self._make_mock_chat_response()

        self.vc.analyze_image(self.test_image)

        called_url = mock_post.call_args[0][0]
        self.assertIn("/api/chat", called_url,
                      f"qwen-vl 應呼叫 /api/chat，但實際呼叫了: {called_url}")
        self.assertNotIn("/api/generate", called_url)

    @patch("vision.vision_connector.requests.post")
    def test_prompt_is_included(self, mock_post):
        """確認自訂 prompt 有正確傳入"""
        mock_post.return_value = self._make_mock_chat_response()
        test_prompt = "What color is the square in this image?"

        self.vc.analyze_image(self.test_image, prompt=test_prompt)

        payload = mock_post.call_args[1].get("json") or mock_post.call_args[0][1]
        user_msg = payload["messages"][0]
        self.assertIn(test_prompt, str(user_msg.get("content", "")),
                      "自訂 prompt 沒有出現在 payload 中")

    @patch("vision.vision_connector.requests.post")
    def test_no_thought_loop_fallback_on_valid_response(self, mock_post):
        """當模型回傳正常內容時，不應觸發 thought loop fallback"""
        expected = "A test image with colored squares."
        mock_post.return_value = self._make_mock_chat_response(expected)

        result = self.vc.analyze_image(self.test_image)

        self.assertEqual(result, expected)
        self.assertNotEqual(result, "A computer screen with active windows.",
                            "不應觸發 thought loop fallback")


class TestAnalyzeImageLiveIntegration(unittest.TestCase):
    """
    🌐 整合測試（需要 Ollama 正在運行）
    執行前請確認：ollama run qwen3-vl:4b-instruct
    若 Ollama 未啟動，這些測試會自動跳過
    """

    @classmethod
    def setUpClass(cls):
        import requests
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            tags = r.json().get("models", [])
            model_names = [m.get("name", "") for m in tags]
            cls.qwen_available = any("qwen3-vl" in n for n in model_names)
            cls.ollama_running = True
        except Exception:
            cls.ollama_running = False
            cls.qwen_available = False

    def setUp(self):
        if not self.ollama_running:
            self.skipTest("Ollama 未啟動，跳過整合測試")
        if not self.qwen_available:
            self.skipTest("qwen3-vl:4b-instruct 尚未下載，執行 `ollama pull qwen3-vl:4b-instruct`")
        self.vc = VisionConnector(model="qwen3-vl:4b-instruct", timeout=60)

    def test_live_image_analysis_returns_content(self):
        """
        ⭐ 真實呼叫：模型必須能描述圖片內容
        如果回傳 'A computer screen with active windows.' 代表圖片沒有傳到模型
        """
        img = make_test_image_pil()
        result = self.vc.analyze_image(
            img,
            prompt="This is a test image. Describe what colors and shapes you can see.",
        )
        print(f"\n[Live Test] Model response: {result}")

        self.assertIsNotNone(result)
        self.assertGreater(len(result), 5)
        # 圖片有確實傳送時，模型不應回傳 fallback 訊息
        self.assertNotEqual(
            result, "A computer screen with active windows.",
            "❌ 模型回傳了 fallback 文字，代表圖片沒有成功傳入！"
        )
        self.assertNotIn(
            "don't have", result.lower(),
            "❌ 模型說沒有收到圖片，payload 格式可能仍然有問題"
        )


if __name__ == "__main__":
    # 顯示詳細輸出
    unittest.main(verbosity=2)
