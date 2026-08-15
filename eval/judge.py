"""Generation scoring.

Two scores, and the distinction between them is the reason this project exists.

**Correctness** compares the generated answer to the expected answer.
**Faithfulness** compares the generated answer to *the retrieved context only*.

A configuration can score high on correctness and low on faithfulness, and that
combination is the interesting failure: the model was right because it already
knew the answer, not because retrieval worked. On an in-house corpus the model
has never seen, that configuration collapses — and a single accuracy number
would never have warned you.

Two backends. ``llm`` is a direct JSON-schema judge (fast, one call, no extra
dependencies). ``ragas`` runs the library's faithfulness / answer_correctness /
context_precision metrics, which is more work per example but gives numbers a
reviewer can compare to published results.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

from src.types import JudgeResult, RetrievalResult

JUDGE_SYSTEM = """You are an expert RAG evaluation judge.

Return ONLY a valid JSON object with exactly this structure:
{"correctness": "correct | partially-correct | incorrect",
 "faithfulness": "faithful | partially-faithful | unfaithful",
 "explanation": "one or two sentences"}

Definitions:
- correctness: does the generated answer match the expected answer in substance?
  Wording may differ. Missing a required part is partially-correct.
- faithfulness: is every factual claim in the generated answer supported by the
  retrieved context below? Judge this using the context alone. A claim that is
  true in the real world but absent from the context makes the answer unfaithful.

Use lowercase enum values exactly as shown. Add no other fields. Return nothing
except the JSON object."""


class Judge(ABC):
    name: str

    @abstractmethod
    def judge(self, query: str, expected: str, generated: str, context: list[RetrievalResult]) -> JudgeResult:
        ...


class LLMJudge(Judge):
    name = "llm-judge"

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("JUDGE_MODEL", os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"))
        self.base_url = base_url or os.environ.get("JUDGE_BASE_URL", os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
        self.api_key = api_key or os.environ.get("JUDGE_API_KEY", os.environ.get("LLM_API_KEY", ""))
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed")
        return self._client

    def judge(self, query: str, expected: str, generated: str, context: list[RetrievalResult]) -> JudgeResult:
        context_text = "\n\n".join(f"[{i}] {r.content}" for i, r in enumerate(context, start=1)) or "(empty)"
        user = (
            f"Question:\n{query}\n\n"
            f"Expected answer:\n{expected}\n\n"
            f"Generated answer:\n{generated}\n\n"
            f"Retrieved context:\n{context_text}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=300,
        )
        return _parse_judge(response.choices[0].message.content or "")


class RagasJudge(Judge):
    """Wraps RAGAS. Heavier, but the metric definitions are somebody else's."""

    name = "ragas"

    def __init__(self, model: str | None = None, embed_model: str = "BAAI/bge-small-en-v1.5"):
        self.model = model or os.environ.get("JUDGE_MODEL", "llama-3.3-70b-versatile")
        self.embed_model = embed_model
        self._ready = False

    def _setup(self):
        if self._ready:
            return
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        self.llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self.model,
                base_url=os.environ.get("JUDGE_BASE_URL", os.environ.get("LLM_BASE_URL")),
                api_key=os.environ.get("JUDGE_API_KEY", os.environ.get("LLM_API_KEY", "not-needed")),
                temperature=0.0,
            )
        )
        self.embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=self.embed_model))
        self._ready = True

    def judge(self, query: str, expected: str, generated: str, context: list[RetrievalResult]) -> JudgeResult:
        self._setup()
        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness, AnswerCorrectness

        sample = SingleTurnSample(
            user_input=query,
            response=generated,
            reference=expected,
            retrieved_contexts=[r.content for r in context],
        )
        faith = Faithfulness(llm=self.llm).single_turn_score(sample)
        corr = AnswerCorrectness(llm=self.llm, embeddings=self.embeddings).single_turn_score(sample)
        return JudgeResult(
            correctness=_bucket(corr, ("incorrect", "partially-correct", "correct")),
            faithfulness=_bucket(faith, ("unfaithful", "partially-faithful", "faithful")),
            explanation=f"ragas answer_correctness={corr:.3f} faithfulness={faith:.3f}",
        )


class StubJudge(Judge):
    """Offline judge for tests. Token overlap, no network."""

    name = "stub"

    def judge(self, query: str, expected: str, generated: str, context: list[RetrievalResult]) -> JudgeResult:
        exp = set(expected.lower().split())
        gen = set(generated.lower().split())
        ctx = set(" ".join(r.content for r in context).lower().split())
        corr = len(exp & gen) / len(exp) if exp else 0.0
        faith = len(gen & ctx) / len(gen) if gen else 0.0
        return JudgeResult(
            correctness=_bucket(corr, ("incorrect", "partially-correct", "correct")),
            faithfulness=_bucket(faith, ("unfaithful", "partially-faithful", "faithful")),
            explanation=f"stub overlap correctness={corr:.2f} faithfulness={faith:.2f}",
        )


def _bucket(score: float, labels: tuple[str, str, str]) -> str:
    if score >= 0.75:
        return labels[2]
    if score >= 0.35:
        return labels[1]
    return labels[0]


def _parse_judge(raw: str) -> JudgeResult:
    """Judges wrap JSON in prose or fences more often than you would like."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return JudgeResult("incorrect", "unfaithful", f"unparseable judge output: {raw[:200]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JudgeResult("incorrect", "unfaithful", f"invalid JSON from judge: {raw[:200]}")

    correctness = str(data.get("correctness", "incorrect")).lower().strip()
    faithfulness = str(data.get("faithfulness", "unfaithful")).lower().strip()
    if correctness not in ("correct", "partially-correct", "incorrect"):
        correctness = "incorrect"
    if faithfulness not in ("faithful", "partially-faithful", "unfaithful"):
        faithfulness = "unfaithful"
    return JudgeResult(correctness, faithfulness, str(data.get("explanation", ""))[:500])


def build_judge(backend: str = "llm") -> Judge:
    if backend == "stub" or os.environ.get("JUDGE_BACKEND") == "stub":
        return StubJudge()
    if backend == "ragas":
        return RagasJudge()
    return LLMJudge()
