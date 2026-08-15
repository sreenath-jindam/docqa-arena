"""Generation.

Anything that speaks the OpenAI chat API works here: Groq's free tier,
OpenRouter, a local Ollama with ``/v1`` enabled, or OpenAI itself. Only the base
URL and model name change, and both come from the environment.

The prompt is deliberately strict about grounding. Faithfulness is one of the
two things being measured, and a generator told "answer from your own knowledge
if the context is thin" would make that measurement meaningless.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from .types import GenerationResult, RetrievalResult

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Rules:
- Use only what the passages state. Do not add facts from your own knowledge.
- If the passages do not contain the answer, say exactly: "The provided context does not contain this information."
- Cite the passage numbers you used, like [2] or [1][3].
- Be concise. Two or three sentences unless the question needs more."""


def format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(f"[{i}] (source: {result.document_id})\n{result.content}")
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def build_messages(query: str, results: list[RetrievalResult]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{format_context(results)}\n\nQuestion: {query}"},
    ]


class Generator(ABC):
    name: str

    @abstractmethod
    def generate(self, messages: list[dict]) -> GenerationResult:
        ...


class OpenAICompatibleGenerator(Generator):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ):
        self.name = model or os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed")
        return self._client

    def generate(self, messages: list[dict]) -> GenerationResult:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = response.usage
        return GenerationResult(
            answer=response.choices[0].message.content or "",
            model=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=latency_ms,
        )


class EchoGenerator(Generator):
    """Offline generator. Returns the top passage verbatim.

    Useful for two things: running tests without an API key, and establishing a
    trivially-faithful floor for the faithfulness metric.
    """

    name = "echo"

    def generate(self, messages: list[dict]) -> GenerationResult:
        start = time.perf_counter()
        user = messages[-1]["content"]
        answer = user.split("[1]", 1)[-1].split("\n\n")[0].strip() if "[1]" in user else "no context"
        return GenerationResult(
            answer=answer[:500],
            model=self.name,
            prompt_tokens=len(user.split()),
            completion_tokens=len(answer.split()),
            total_tokens=len(user.split()) + len(answer.split()),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def build_generator(model: str | None = None) -> Generator:
    if model == "echo" or os.environ.get("LLM_BACKEND") == "echo":
        return EchoGenerator()
    return OpenAICompatibleGenerator(model=model)
