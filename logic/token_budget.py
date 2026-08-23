"""Prompt rendering and conservative context budgeting for the Meguru model."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from .context_manifest import parameter_int
from .prompt_boundaries import USER_MESSAGE_MARKER


CONTEXT_WINDOW_TOKENS = 4096
MAX_OUTPUT_TOKENS = 300
SAFETY_MARGIN_TOKENS = 128
MAX_PROMPT_TOKENS = (
    CONTEXT_WINDOW_TOKENS - MAX_OUTPUT_TOKENS - SAFETY_MARGIN_TOKENS
)
MAX_USER_TOKENS = 768
TOKENIZER_VALIDATION_TOLERANCE_TOKENS = 8

_TEMPLATE_MARKERS = (
    "<|im_start|>system",
    "<|im_start|>user",
    "<|im_start|>assistant",
    "<|im_end|>",
)


class PromptBudgetError(ValueError):
    """A request cannot be represented safely within the model context."""

    def __init__(self, code: str, **details: Any) -> None:
        self.code = code
        self.details = details
        super().__init__(code)


class PromptProfileError(ValueError):
    """The active Ollama prompt template is not one we can count exactly."""


@dataclass(frozen=True)
class PromptProfile:
    """The system prompt and supported ChatML rendering used by Meguru."""

    system: str
    source: str
    template: str = "meguru_chatml"
    user_suffix: str = ""
    verified: bool = False
    num_ctx: int | None = None
    num_predict: int | None = None
    parameters: str = ""

    def render(self, prompt: str) -> str:
        # This mirrors tools/model_training/Modelfile. The response variable is
        # intentionally omitted because Ollama begins generation at the
        # assistant prefix.
        result = ""
        if self.system:
            result += f"<|im_start|>system\n{self.system}<|im_end|>\n"
        if prompt:
            result += f"<|im_start|>user\n{prompt}{self.user_suffix}<|im_end|>\n"
        return result + "<|im_start|>assistant\n"

    @classmethod
    def from_modelfile(cls, path: Path | None = None) -> "PromptProfile":
        path = path or _default_modelfile_path()
        text = path.read_text(encoding="utf-8")
        match = re.search(r"SYSTEM\s+\"\"\"(.*?)\"\"\"", text, re.DOTALL)
        if not match:
            raise PromptProfileError(f"SYSTEM prompt not found in {path}")
        suffix = "\n/no_think" if "/no_think" in text else ""
        return cls(
            system=match.group(1),
            source=str(path),
            user_suffix=suffix,
            verified=False,
            num_ctx=parameter_int(text, "num_ctx"),
            num_predict=parameter_int(text, "num_predict"),
        )

    @classmethod
    def from_ollama_response(
        cls, payload: Mapping[str, Any], source: str = "ollama:/api/show"
    ) -> "PromptProfile":
        template = str(payload.get("template") or "")
        if not _is_supported_template(template):
            raise PromptProfileError(
                "Ollama model uses an unsupported prompt template; "
                "exact prompt accounting is disabled"
            )
        system = str(payload.get("system") or "")
        suffix = "\n/no_think" if "/no_think" in template else ""
        parameters = str(payload.get("parameters") or "")
        return cls(
            system=system,
            source=source,
            user_suffix=suffix,
            verified=True,
            num_ctx=parameter_int(parameters, "num_ctx"),
            num_predict=parameter_int(parameters, "num_predict"),
            parameters=parameters,
        )


def load_prompt_profile(
    *,
    endpoint: str | None = None,
    model: str | None = None,
    modelfile: Path | None = None,
    timeout: float = 1.0,
) -> PromptProfile:
    """Load the active Ollama profile, falling back to the checked-in model file."""

    return load_prompt_runtime(
        endpoint=endpoint,
        model=model,
        modelfile=modelfile,
        timeout=timeout,
    )[0]


def load_prompt_runtime(
    *,
    endpoint: str | None = None,
    model: str | None = None,
    modelfile: Path | None = None,
    timeout: float = 1.0,
) -> tuple[PromptProfile, Mapping[str, Any] | None]:
    """Return the active profile and its authoritative show payload once."""

    endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "meguru")
    try:
        request = Request(
            f"{endpoint.rstrip('/')}/api/show",
            data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return PromptProfile.from_ollama_response(payload), payload
    except (OSError, URLError, ValueError, PromptProfileError):
        # Startup should not make ordinary chat dependent on a live Ollama
        # inspection call. The active model is still checked by verification.
        return PromptProfile.from_modelfile(modelfile), None


def _default_modelfile_path() -> Path:
    return Path(__file__).resolve().parent.parent / "tools" / "model_training" / "Modelfile"


def _is_supported_template(template: str) -> bool:
    compact = re.sub(r"\s+", "", template)
    return all(marker.replace(" ", "") in compact for marker in _TEMPLATE_MARKERS) and all(
        token in compact for token in (".System", ".Prompt")
    )


class TokenCounter:
    """Count with the exact GGUF tokenizer when available, else use a safe bound.

    The retained training tokenizer is not available, so an official Qwen
    tokenizer is deliberately not treated as interchangeable.  The optional
    ``llama-cpp-python`` path loads the vocabulary directly from the retained
    GGUF.  If it is not installed (or cannot load the file), the byte counter
    keeps requests safe and reports that RAG should remain disabled.
    """

    def __init__(self, tokenizer: Any = None, mode: str = "utf8_upper_bound") -> None:
        self._tokenizer = tokenizer
        self.mode = mode

    @property
    def exact(self) -> bool:
        return self.mode == "gguf_native"

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            if hasattr(self._tokenizer, "tokenize"):
                data = text.encode("utf-8")
                try:
                    return len(
                        self._tokenizer.tokenize(
                            data, add_bos=False, special=True
                        )
                    )
                except TypeError:
                    return len(self._tokenizer.tokenize(data, add_bos=False))
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        # A byte cannot require more than one byte-level token. The extra
        # allowance covers renderer/BOS accounting when this is used on a
        # complete rendered prompt.
        return len(text.encode("utf-8")) + 32

    @classmethod
    def load(cls, path: str | Path | None = None) -> "TokenCounter":
        mode = os.getenv("MEGURU_TOKENIZER_MODE", "auto").lower()
        if mode in {"utf8", "utf8_upper_bound", "disabled"}:
            return cls()

        path = Path(
            path
            or os.getenv(
                "MEGURU_GGUF_PATH",
                "tools/model_training/meguru_q4_k_m.gguf",
            )
        )
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if not path.exists():
            return cls()

        try:
            from llama_cpp import Llama

            tokenizer = Llama(
                model_path=str(path),
                vocab_only=True,
                n_ctx=512,
                verbose=False,
            )
            return cls(tokenizer=tokenizer, mode="gguf_native")
        except Exception:
            return cls()


@dataclass(frozen=True)
class PromptSection:
    name: str
    content: str


@dataclass(frozen=True)
class BudgetedPrompt:
    injected: str
    rendered: str
    user_tokens: int
    base_tokens: int
    final_tokens: int
    counter_mode: str
    dropped_sections: tuple[str, ...]
    knowledge_included: int
    knowledge_dropped: int
    profile_source: str
    profile_verified: bool


class PromptBuilder:
    """Build the current single-message prompt and enforce Phase 1 limits."""

    _DROP_ORDER = ("time", "holiday", "last_topic", "name")

    def __init__(
        self,
        *,
        profile: PromptProfile | None = None,
        counter: TokenCounter | None = None,
        prompt_limit: int = MAX_PROMPT_TOKENS,
        max_user_tokens: int = MAX_USER_TOKENS,
    ) -> None:
        self.profile = profile or load_prompt_profile()
        self.counter = counter or TokenCounter.load()
        self.prompt_limit = prompt_limit
        self.max_user_tokens = max_user_tokens

    def build(
        self,
        user_text: str,
        sections: Sequence[PromptSection] = (),
        knowledge_blocks: Sequence[str] = (),
    ) -> BudgetedPrompt:
        user_text = user_text.strip()
        user_tokens = self.counter.count(user_text)
        if user_tokens > self.max_user_tokens:
            raise PromptBudgetError(
                "message_exceeds_user_limit",
                limit_tokens=self.max_user_tokens,
                measured_tokens=user_tokens,
                counter_mode=self.counter.mode,
            )

        selected = {section.name: section for section in sections if section.content}
        dropped: list[str] = []

        def render_selected(
            extra_knowledge: Sequence[str] = (),
        ) -> tuple[str, str, int]:
            # Keep every retrieved block in the context section.  The user
            # utterance is deliberately rendered last so retrieved text can
            # never become the final semantic instruction in the prompt.
            context_parts = [section.content for section in selected.values()]
            context_parts.extend(extra_knowledge)
            context_hint = "\n".join(context_parts)
            injected = (
                f"{context_hint}\n\n{USER_MESSAGE_MARKER} {user_text}"
                if context_hint
                else user_text
            )
            rendered = self.profile.render(injected)
            return injected, rendered, self.counter.count(rendered)

        injected, rendered, base_tokens = render_selected()
        for section_name in self._DROP_ORDER:
            if base_tokens <= self.prompt_limit:
                break
            if section_name in selected:
                selected.pop(section_name)
                dropped.append(section_name)
                injected, rendered, base_tokens = render_selected()

        if base_tokens > self.prompt_limit:
            raise PromptBudgetError(
                "message_exceeds_total_context_budget",
                prompt_limit_tokens=self.prompt_limit,
                measured_prompt_tokens=base_tokens,
                counter_mode=self.counter.mode,
                dropped_sections=tuple(dropped),
            )

        knowledge_included = 0
        knowledge_dropped = 0
        included_blocks: list[str] = []
        # Retrieved knowledge is added only as complete blocks before the user
        # marker. This keeps each source self-contained and lets ordinary chat
        # continue when no block fits the remaining budget.
        for block in knowledge_blocks:
            if not block:
                continue
            candidate_blocks = [*included_blocks, block]
            candidate, candidate_rendered, candidate_tokens = render_selected(
                candidate_blocks
            )
            if candidate_tokens <= self.prompt_limit:
                injected = candidate
                rendered = candidate_rendered
                included_blocks.append(block)
                knowledge_included += 1
            else:
                knowledge_dropped += 1

        return BudgetedPrompt(
            injected=injected,
            rendered=rendered,
            user_tokens=user_tokens,
            base_tokens=base_tokens,
            final_tokens=self.counter.count(rendered),
            counter_mode=self.counter.mode,
            dropped_sections=tuple(dropped),
            knowledge_included=knowledge_included,
            knowledge_dropped=knowledge_dropped,
            profile_source=self.profile.source,
            profile_verified=self.profile.verified,
        )


def current_context_sections(
    *, holiday_hint: str, time_text: str, user_name: str, last_topic: str
) -> tuple[PromptSection, ...]:
    """Preserve the existing labels while making them budgetable sections."""

    sections: list[PromptSection] = []
    if holiday_hint:
        sections.append(PromptSection("holiday", f"[今日のイベント: {holiday_hint}]"))
    if time_text:
        sections.append(PromptSection("time", f"[現在の時間帯: {time_text}]"))
    if user_name:
        sections.append(PromptSection("name", f"[ユーザー名: {user_name}]"))
    if last_topic:
        sections.append(PromptSection("last_topic", f"[前回の話題: {last_topic}]"))
    return tuple(sections)
