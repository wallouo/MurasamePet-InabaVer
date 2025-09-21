"""
自動化測試：/say 端點。
測試輸入 text 時會生成 wav 並返回字幕。
"""

import os
import unittest

from fastapi.testclient import TestClient

import api as api_module


class SayApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_module.app)

    def test_say_generates_wav_and_subtitle(self):
        # force mock
        if 'VOICEVOX_ENDPOINT' in os.environ:
            del os.environ['VOICEVOX_ENDPOINT']
        res = self.client.post('/say', json={"text": "測試一下"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('wav_path', data)
        wav_path = data['wav_path']
        self.assertTrue(os.path.exists(wav_path))
        subtitle = data.get('subtitle_zh', '')
        self.assertTrue(len(subtitle) > 0)


if __name__ == '__main__':
    unittest.main()