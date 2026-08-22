import unittest

from logic.token_budget import (
    PromptBudgetError,
    PromptBuilder,
    PromptProfile,
    PromptSection,
    TokenCounter,
    current_context_sections,
)


class CharacterCounter(TokenCounter):
    """Deterministic counter for boundary tests without model downloads."""

    def __init__(self):
        super().__init__(mode="test")

    def count(self, text: str) -> int:
        return len(text)


class TokenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.profile = PromptProfile(system="meguru", source="test")
        self.counter = CharacterCounter()

    def test_user_limit_has_stable_error_code(self):
        builder = PromptBuilder(
            profile=self.profile,
            counter=self.counter,
            prompt_limit=10_000,
            max_user_tokens=768,
        )

        with self.assertRaises(PromptBudgetError) as raised:
            builder.build("x" * 769)

        self.assertEqual(raised.exception.code, "message_exceeds_user_limit")
        self.assertEqual(raised.exception.details["limit_tokens"], 768)

    def test_total_context_limit_has_distinct_error_code(self):
        builder = PromptBuilder(
            profile=PromptProfile(system="s" * 100, source="test"),
            counter=self.counter,
            prompt_limit=50,
            max_user_tokens=768,
        )

        with self.assertRaises(PromptBudgetError) as raised:
            builder.build("hello")

        self.assertEqual(
            raised.exception.code,
            "message_exceeds_total_context_budget",
        )

    def test_existing_context_labels_are_preserved(self):
        builder = PromptBuilder(profile=self.profile, counter=self.counter)
        result = builder.build(
            "hello",
            current_context_sections(
                holiday_hint="イベント",
                time_text="朝",
                user_name="センパイ",
                last_topic="ゲーム",
            ),
        )

        self.assertIn("[今日のイベント: イベント]", result.injected)
        self.assertIn("[現在の時間帯: 朝]", result.injected)
        self.assertIn("[ユーザー名: センパイ]", result.injected)
        self.assertIn("[前回の話題: ゲーム]", result.injected)
        self.assertIn("ユーザーの発言: hello", result.injected)

    def test_active_qwen_no_think_suffix_is_counted(self):
        profile = PromptProfile.from_ollama_response(
            {
                "system": "meguru",
                "parameters": "num_ctx 4096\nnum_predict 300",
                "template": (
                    "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n"
                    "{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}"
                    "\n/no_think<|im_end|>\n{{ end }}<|im_start|>assistant\n"
                ),
            }
        )

        rendered = profile.render("hello")

        self.assertIn("hello\n/no_think<|im_end|>", rendered)
        self.assertTrue(profile.verified)
        self.assertEqual(profile.num_ctx, 4096)
        self.assertEqual(profile.num_predict, 300)

    def test_missing_gguf_uses_conservative_counter(self):
        counter = TokenCounter.load("does-not-exist.gguf")

        self.assertEqual(counter.mode, "utf8_upper_bound")
        self.assertFalse(counter.exact)

    def test_low_priority_sections_are_dropped_first(self):
        builder = PromptBuilder(
            profile=PromptProfile(system="", source="test"),
            counter=self.counter,
            prompt_limit=86,
        )
        sections = (
            PromptSection("holiday", "holiday"),
            PromptSection("time", "time"),
            PromptSection("name", "name"),
            PromptSection("last_topic", "last_topic"),
        )

        result = builder.build("hello", sections)

        self.assertEqual(result.dropped_sections[:2], ("time", "holiday"))
        self.assertIn("name", result.injected)
        self.assertIn("last_topic", result.injected)


if __name__ == "__main__":
    unittest.main()
