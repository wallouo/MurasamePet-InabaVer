import unittest
from dataclasses import replace

from logic.knowledge import KnowledgeSearchResult, ROUTE_SCOPES
from logic.rag import KnowledgeRetriever, RAGSettings, format_knowledge_block


class _FakeStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, *, limit, min_similarity, route_scopes=None):
        self.calls.append(
            {
                "query": query,
                "limit": limit,
                "min_similarity": min_similarity,
                "route_scopes": route_scopes,
            }
        )
        return list(self.results)


class RagTests(unittest.TestCase):
    def setUp(self):
        self.result = KnowledgeSearchResult(
            chunk_id="chunk-1",
            text="因幡めぐるはゲームが好きで、センパイと同じ学園に通っている。",
            metadata={
                "document_id": "meguru-profile",
                "title": "Meguru profile",
                "source_url": "https://example.test/meguru",
                "document_format": "markdown",
                "source_authority": "curated",
                "route_scope": "meguru",
                "perspective_status": "lived",
                "language": "ja",
                "character_tags": '["meguru", "protagonist"]',
                "relationship_tags": '["senpai"]',
            },
            distance=0.1,
            similarity=0.9,
        )

    def test_format_preserves_grounding_metadata_and_content(self):
        block = format_knowledge_block(self.result)

        self.assertIn("document_id: meguru-profile", block)
        self.assertIn("source_authority: curated", block)
        self.assertIn("character_tags: meguru, protagonist", block)
        self.assertIn("relationship_tags: senpai", block)
        self.assertIn("因幡めぐるはゲームが好き", block)
        self.assertIn("参考知識ここまで", block)

    def test_retriever_passes_route_and_relevance_settings(self):
        settings = RAGSettings(
            max_results=2,
            min_similarity=0.78,
            route_scope="meguru",
        )
        store = _FakeStore([self.result])
        retriever = KnowledgeRetriever(settings, store=store)

        results = retriever.search("めぐるの設定")

        self.assertEqual(results, [self.result])
        self.assertEqual(
            store.calls,
            [
                {
                    "query": "めぐるの設定",
                    "limit": 2 * len(ROUTE_SCOPES),
                    "min_similarity": 0.78,
                    "route_scopes": None,
                }
            ],
        )

    def test_all_routes_are_searchable_and_preference_only_breaks_ties(self):
        settings = RAGSettings(max_results=1, route_scope="meguru")
        nene = replace(
            self.result,
            chunk_id="nene-1",
            similarity=0.91,
            metadata={**self.result.metadata, "route_scope": "nene"},
        )
        meguru = replace(self.result, similarity=0.90)
        store = _FakeStore([meguru, nene])

        results = KnowledgeRetriever(settings, store=store).search("route fact")

        self.assertEqual([item.metadata["route_scope"] for item in results], ["nene"])
        self.assertIsNone(store.calls[0]["route_scopes"])

        nene_tie = replace(nene, similarity=0.90)
        tie_store = _FakeStore([nene_tie, meguru])
        tie_results = KnowledgeRetriever(settings, store=tie_store).search("route fact")
        self.assertEqual(
            [item.metadata["route_scope"] for item in tie_results], ["meguru"]
        )

    def test_alternate_route_is_explicitly_labelled(self):
        alternate = replace(
            self.result,
            metadata={
                **self.result.metadata,
                "route_scope": "nene",
                "perspective_status": "alternate",
            },
        )

        block = format_knowledge_block(alternate)

        self.assertIn("perspective_status: alternate", block)
        self.assertIn("alternate-route knowledge", block)
        self.assertIn("do not present as Meguru and Senpai's lived history", block)

    def test_retriever_degrades_to_empty_results_on_store_failure(self):
        class BrokenStore:
            def search(self, **kwargs):
                raise RuntimeError("index unavailable")

        retriever = KnowledgeRetriever(
            RAGSettings(),
            store=BrokenStore(),
        )

        self.assertEqual(retriever.search("anything"), [])
        self.assertIn("index unavailable", retriever.last_error)


if __name__ == "__main__":
    unittest.main()
