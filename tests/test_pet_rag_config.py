import unittest
from unittest.mock import Mock, patch

import pet


class DesktopRAGConfigurationTests(unittest.TestCase):
    def test_chat_worker_defaults_to_no_knowledge(self):
        worker = pet.ChatWorker("hello", "master", "http://127.0.0.1:5000")
        response = Mock()
        response.json.return_value = {}
        with patch("requests.post", return_value=response) as post:
            worker.run()
        self.assertFalse(post.call_args.kwargs["json"]["use_knowledge"])

    def test_chat_worker_can_request_knowledge(self):
        worker = pet.ChatWorker(
            "hello",
            "master",
            "http://127.0.0.1:5000",
            use_knowledge=True,
        )
        response = Mock()
        response.json.return_value = {}
        with patch("requests.post", return_value=response) as post:
            worker.run()
        self.assertTrue(post.call_args.kwargs["json"]["use_knowledge"])


if __name__ == "__main__":
    unittest.main()
