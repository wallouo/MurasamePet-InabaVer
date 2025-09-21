"""
自動化測試：/pat 端點。

驗證當 VOICEVOX 不可達時會自動產生 mock wav，且返回的 wav 路徑存在，字幕非空。
"""

import os
import unittest

from fastapi.testclient import TestClient

import api as api_module


class PatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_module.app)

    def test_pat_generates_wav_and_subtitle(self):
        # 移除 VOICEVOX 以強制使用 mock
        if 'VOICEVOX_ENDPOINT' in os.environ:
            del os.environ['VOICEVOX_ENDPOINT']
        res = self.client.post('/pat')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn('wav_path', data)
        wav_path = data['wav_path']
        self.assertTrue(os.path.exists(wav_path))
        subtitle = data.get('subtitle_zh', '')
        self.assertTrue(len(subtitle) > 0)


if __name__ == '__main__':
    unittest.main()