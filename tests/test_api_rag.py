import unittest
from unittest.mock import AsyncMock, patch

import api as api_module
from logic.knowledge import KnowledgeSearchResult
from logic.rag import KnowledgeRetriever, RAGSettings
from logic.token_budget import PromptBuilder, PromptProfile, TokenCounter


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "うん、知ってるよ♪"}}


class _Retriever(KnowledgeRetriever):
    def __init__(self, result):
        super().__init__(RAGSettings(), store=None)
        self.result = result
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        return [self.result]


class _CharacterCounter(TokenCounter):
    def __init__(self):
        super().__init__(mode="test")

    def count(self, text):
        return len(text)


class ApiRagTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.result = KnowledgeSearchResult(
            chunk_id="meguru-1",
            text="因幡めぐるはゲームが好き。",
            metadata={
                "document_id": "meguru-profile",
                "title": "Meguru",
                "document_format": "markdown",
                "source_authority": "curated",
                "route_scope": "meguru",
                "perspective_status": "lived",
                "language": "ja",
            },
            distance=0.1,
            similarity=0.9,
        )
        self.retriever = _Retriever(self.result)

    async def test_opt_in_retrieval_is_injected_into_ollama_prompt(self):
        with (
            patch.object(api_module, "RAG_ENABLED", True),
            patch.object(api_module, "_rag_retriever", self.retriever),
            patch.object(
                api_module,
                "_prompt_builder",
                PromptBuilder(
                    profile=PromptProfile(system="", source="test"),
                    counter=_CharacterCounter(),
                    prompt_limit=1_000,
                ),
            ),
            patch.object(api_module._memory, "get", return_value={"name": "", "last_topic": "", "mood": ""}),
            patch.object(api_module._memory, "set"),
            patch.object(api_module.requests, "post", return_value=_Response()) as post,
            patch.object(
                api_module,
                "tts",
                new=AsyncMock(return_value={"wav_path": "", "backend": "mock"}),
            ),
        ):
            response = await api_module.chat_process(
                api_module.UserChatRequest(text="めぐるの設定を教えて", use_knowledge=True)
            )

        self.assertEqual(self.retriever.calls, ["めぐるの設定を教えて"])
        prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("document_id: meguru-profile", prompt)
        self.assertIn("因幡めぐるはゲームが好き", prompt)
        self.assertTrue(response["text"])

    async def test_default_request_does_not_retrieve_knowledge(self):
        with (
            patch.object(api_module, "RAG_ENABLED", True),
            patch.object(api_module, "_rag_retriever", self.retriever),
            patch.object(api_module._memory, "get", return_value={"name": "", "last_topic": "", "mood": ""}),
            patch.object(api_module._memory, "set"),
            patch.object(api_module.requests, "post", return_value=_Response()),
            patch.object(
                api_module,
                "tts",
                new=AsyncMock(return_value={"wav_path": "", "backend": "mock"}),
            ),
        ):
            await api_module.chat_process(api_module.UserChatRequest(text="こんにちは"))

        self.assertEqual(self.retriever.calls, [])


if __name__ == "__main__":
    unittest.main()
