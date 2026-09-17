"""Layer 2: RAGAS metrics over the same dataset.

Retrieved guide chunks are pulled out of the trace, so RAGAS scores the contexts
the agent actually used rather than a re-run of retrieval.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from evals.deterministic import EvalCase
from tripmate.core.trace import EventType
from tripmate.models import AgentResponse

PROVIDER_OPENAI_COMPATIBLE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": None,  # the default; ChatOpenAI needs no base_url override
    "together_ai": "https://api.together.xyz/v1",
    "mistral": "https://api.mistral.ai/v1",
}

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_without_reference",
)


def _contexts_from(response: AgentResponse) -> list[str]:
    """Guide text the agent retrieved this turn, taken from the trace."""
    contexts: list[str] = []
    for event in response.trace:
        if event.event_type != EventType.TOOL_RESULT:
            continue
        if event.payload.get("tool") != "search_destination_guide":
            continue
        contexts.extend(event.payload.get("texts", []))
    return contexts


def _judge_llm():
    """The LLM RAGAS uses to grade answers.

    RAGAS defaults to OpenAI and raises without OPENAI_API_KEY. Rather than demand a
    second provider, point its OpenAI-compatible client at whatever provider the project
    is already configured for — Groq exposes an OpenAI-shaped endpoint, so the same key
    that runs the agent also grades it.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from tripmate.config import get_settings

    settings = get_settings()
    settings.export_provider_key()
    provider = settings.provider
    base_url = settings.llm_base_url or PROVIDER_OPENAI_COMPATIBLE_URLS.get(provider)
    if base_url is None:
        raise RuntimeError(
            f"No OpenAI-compatible base URL known for provider {provider!r}. "
            f"Set LLM_BASE_URL, or run the RAGAS layer against an OpenAI-compatible provider."
        )

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.llm_model.split("/", 1)[-1],
            base_url=base_url,
            api_key=settings.llm_api_key,
            temperature=0.0,
        )
    )


def _judge_embeddings():
    """Local ONNX embeddings for answer_relevancy — no embedding API key needed."""
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from tripmate.config import get_settings

    return LangchainEmbeddingsWrapper(
        FastEmbedEmbeddings(model_name=get_settings().embedding_model)
    )


def collect_samples(agent: Any, cases: list[EvalCase]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for case in cases:
        if "search_destination_guide" not in case.expected_tools:
            continue
        response = agent.chat(case.query, session_id=case.session_id)
        contexts = _contexts_from(response)
        if not contexts:
            continue
        samples.append({
            "question": case.query,
            "answer": response.answer,
            "contexts": contexts,
        })
    return samples


def run_ragas(agent: Any, cases: list[EvalCase]) -> dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        LLMContextPrecisionWithoutReference,
        answer_relevancy,
        faithfulness,
    )

    samples = collect_samples(agent, cases)
    if not samples:
        return {}

    # context_precision requires a `reference` (ground-truth answer) column. This
    # dataset has none — hand-authoring 30 reference answers is its own project, and a
    # weak one would score worse than no score. The reference-free variant measures the
    # same thing from the question and retrieved contexts alone.
    result = evaluate(
        Dataset.from_list(samples),
        metrics=[
            faithfulness,
            answer_relevancy,
            LLMContextPrecisionWithoutReference(),
        ],
        llm=_judge_llm(),
        embeddings=_judge_embeddings(),
    )
    # EvaluationResult is not a mapping — `name in result` hits __getitem__ with an
    # int. to_pandas() gives one row per sample with a column per metric; the mean over
    # those rows is the score for the run.
    frame = result.to_pandas()
    return {
        name: float(frame[name].mean())
        for name in METRIC_NAMES
        if name in frame.columns and frame[name].notna().any()
    }


def print_ragas_report(console: Console, scores: dict[str, float]) -> None:
    if not scores:
        console.print("[yellow]RAGAS: no RAG samples collected[/yellow]")
        return
    table = Table(title="RAGAS metrics", show_header=False)
    for name, value in scores.items():
        table.add_row(name.replace("_", " "), f"{value:.3f}")
    console.print(table)
