import unittest
from types import SimpleNamespace
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

    def test_normal_typed_chat_forwards_only_the_explicit_desktop_setting(self):
        for configured in (False, True):
            with self.subTest(configured=configured):
                chat_input = Mock()
                chat_input.text.return_value = "hello"
                fake_window = SimpleNamespace(
                    chat_input=chat_input,
                    subtitle=Mock(),
                    geometry=Mock(return_value=None),
                    handle_api_response=Mock(),
                )
                worker = Mock()
                worker.finished = Mock()
                worker.error = Mock()
                with (
                    patch.object(pet, "RAG_ENABLED", configured),
                    patch.object(pet, "ChatWorker", return_value=worker) as worker_type,
                ):
                    pet.PetWindow.send_chat_message(fake_window)

                worker_type.assert_called_once_with(
                    "hello",
                    "master",
                    pet.API_URL,
                    use_knowledge=configured,
                )
                worker.start.assert_called_once_with()

    def test_pat_action_does_not_send_chat_knowledge_configuration(self):
        response = Mock()
        response.json.return_value = {}
        fake_window = SimpleNamespace(
            handle_api_response=Mock(),
            subtitle=Mock(),
            geometry=Mock(return_value=None),
        )
        with patch.object(pet.requests, "post", return_value=response) as post:
            pet.PetWindow.trigger_pat(fake_window)

        self.assertEqual(post.call_args.args[0], f"{pet.API_URL}/pat")
        self.assertNotIn("json", post.call_args.kwargs)

    def test_vision_comment_does_not_opt_into_knowledge(self):
        response = Mock()
        response.json.return_value = {}
        with patch.object(pet.requests, "post", return_value=response) as post:
            pet.PetWindow._generate_comment(object(), "a game screen")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(set(payload), {"text", "user_id"})
        self.assertNotIn("use_knowledge", payload)


if __name__ == "__main__":
    unittest.main()
