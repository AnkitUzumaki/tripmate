# TripMate Agentic AI Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the agentic core of TripMate — an AI travel assistant that dynamically decides which tools to call per query, chains a RAG tool and a weather tool when needed, synthesizes one grounded answer with citations, and emits a visible reasoning trace.

**Architecture:** A raw LLM function-calling loop (no agent framework) drives a decorator-based tool registry whose JSON schemas are derived from Python type hints. Two tools sit behind it: a RAG tool over ChromaDB with city metadata pre-filtering, and a weather tool over Open-Meteo with a forecast path, a climate-normal path, and a mock fallback. CLI and FastAPI are thin adapters over one `Agent` object; traces are append-only JSONL persisted alongside sessions in SQLAlchemy.

**Tech Stack:** Python 3.10+, uv, LiteLLM, ChromaDB, fastembed, pydantic v2, pydantic-settings, SQLAlchemy 2.0, FastAPI, httpx, diskcache, structlog, rich, pytest, respx, RAGAS.

**Spec:** `docs/superpowers/specs/2026-09-16-tripmate-agent-design.md`

---

## Global Constraints

- **Python 3.10+.** Use `X | None` union syntax, not `Optional[X]`.
- **Package manager: `uv`.** All installs are `uv add`; all test runs are `uv run pytest`.
- **No hardcoded secrets.** Every credential and tunable comes from `Settings` in `src/tripmate/config.py`. A literal API key, URL, threshold, or timeout anywhere outside `config.py` is a defect.
- **No mutation.** Build new objects; never modify inputs in place. Pydantic models use `model_copy(update=...)`.
- **File size:** target 120 lines, hard ceiling 200. Function size: 50 lines max.
- **Nesting:** maximum 4 levels. Use early returns.
- **Tools never raise into the agent loop.** Every tool returns a `ToolResult`; exceptions are caught at the registry boundary and converted to `ToolResult(status="error", ...)`.
- **Naming:** functions/variables `snake_case`, classes/models `PascalCase`, constants `UPPER_SNAKE_CASE`, booleans prefixed `is_`/`has_`/`should_`/`can_`.
- **Scoring convention:** similarity is always `1 - cosine_distance`, expressed in `[0, 1]`. Chroma returns distance; `ChromaStore` converts before any comparison.
- **Commit format:** `<type>: <description>` where type is one of `feat|fix|refactor|docs|test|chore|perf|ci`. No attribution trailers.
- **TDD is mandatory.** Every task writes the failing test first, watches it fail, then implements.
- **Coverage floor: 80%** line coverage, enforced in the final task.

### Git identity note

This repository must **not** be committed under the work account configured globally on this machine. Task 1 initialises the repo and sets a **repository-local** identity. Substitute the personal name and email before running those commands. A remote is added later, by hand, when the personal GitHub account is ready.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, pytest and coverage configuration |
| `.env.example` | Every environment variable with a safe default |
| `src/tripmate/config.py` | `Settings`; maps `LLM_API_KEY` to the provider-specific env var |
| `src/tripmate/models.py` | `Chunk`, `WeatherReport`, `ToolResult`, `Citation`, `TraceEvent`, `AgentResponse`, `LLMResponse`, `ToolCall` |
| `src/tripmate/db.py` | SQLAlchemy engine, session factory, ORM tables |
| `src/tripmate/tools/registry.py` | `@tool` decorator, schema derivation, validating dispatch |
| `src/tripmate/tools/destination.py` | `search_destination_guide` |
| `src/tripmate/tools/weather.py` | `get_weather_forecast`, Open-Meteo client, mock fallback |
| `src/tripmate/rag/chunker.py` | Section-aware splitting, stable chunk IDs |
| `src/tripmate/rag/store.py` | `VectorStore` Protocol, `ChromaStore` |
| `src/tripmate/rag/ingest.py` | Idempotent ingest entry point |
| `src/tripmate/llm/client.py` | LiteLLM wrapper: retries, token and cost extraction |
| `src/tripmate/core/prompts.py` | Versioned system prompt |
| `src/tripmate/core/trace.py` | `Tracer`: structured events, JSONL sink, cost/latency totals |
| `src/tripmate/core/cache.py` | Semantic query cache |
| `src/tripmate/core/agent.py` | The orchestration loop |
| `src/tripmate/adapters/cli.py` | Multi-turn REPL with live rich trace rendering |
| `src/tripmate/adapters/api.py` | FastAPI routes |
| `tests/fakes.py` | `FakeLLMClient` — scripted tool-call replay |
| `evals/` | Dataset plus deterministic, RAGAS, and simulation scorers |
| `Dockerfile` | Multi-stage build |

---

## Task Sequence

| # | Task | Deliverable |
|---|---|---|
| 1 | Scaffold, config, git | `Settings` loads, validates, maps provider keys |
| 2 | Domain models | All pydantic models round-trip |
| 3 | Tool registry | `@tool` derives schemas; dispatch validates |
| 4 | Chunker | 4 files to 20 stable chunks |
| 5 | Vector store + ingest | Semantic search with city filter and score floor |
| 6 | Destination tool | `search_destination_guide` returns `ToolResult` |
| 7 | Weather tool | Both paths, cache, fallback |
| 8 | LLM client + fake | Real client plus scripted `FakeLLMClient` |
| 9 | Tracer | Structured events to JSONL, totals |
| 10 | Persistence | Sessions, turns, trace records, cache rows |
| 11 | Semantic cache | Cosine replay above threshold |
| 12 | Agent core | The loop: selection, parallel dispatch, citations |
| 13 | Tool-selection tests | Single, multi, none |
| 14 | Error scenario tests | All six brief scenarios |
| 15 | Integration test | Full multi-tool flow |
| 16 | CLI adapter | REPL with live trace |
| 17 | FastAPI adapter | `/chat`, `/health`, `/sessions/{id}` |
| 18 | Eval harness part 1 | Dataset plus deterministic scoring |
| 19 | Eval harness part 2 | RAGAS plus simulated user |
| 20 | Packaging | Dockerfile, README, diagram, captured traces, coverage |

---

### Task 1: Project scaffold and configuration

**Files:**
- Create: `pyproject.toml`, `.env.example`, `.gitignore`, `src/tripmate/__init__.py`, `src/tripmate/config.py`, `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `Settings` (pydantic-settings `BaseSettings`) with the fields listed in Step 5
  - `Settings.provider -> str` — provider slug parsed from `llm_model`
  - `Settings.export_provider_key() -> None` — writes `llm_api_key` into the provider-specific env var; raises `ConfigError` when a required key is absent
  - `get_settings() -> Settings` — cached accessor used everywhere else
  - `ConfigError(Exception)`

- [ ] **Step 1: Create the project skeleton and initialise git with a repository-local identity**

Replace the name and email with the personal account details before running.

```bash
cd tripmate
mkdir -p src/tripmate/{tools,rag,llm,core,adapters} tests evals data/destinations traces docs
touch src/tripmate/__init__.py tests/__init__.py
touch src/tripmate/{tools,rag,llm,core,adapters}/__init__.py

git init
git config user.name  "Ankit Singh Chauhan"
git config user.email "PERSONAL_EMAIL_HERE"
git config --local --list | grep user
```

Expected: the two `user.*` lines echo the personal identity, not the work account.

- [ ] **Step 2: Copy the destination data pack into the repository**

```bash
cp ~/Documents/tripmate-destination-data/*.txt data/destinations/ 2>/dev/null \
  || unzip -j ~/Documents/tripmate-destination-data.zip '*.txt' -d data/destinations/
ls data/destinations/
```

Expected: `bangkok.txt barcelona.txt README.txt reykjavik.txt tokyo.txt`

```bash
rm data/destinations/README.txt
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "tripmate"
version = "0.1.0"
description = "Agentic AI travel assistant with RAG and weather tools"
requires-python = ">=3.10"
dependencies = [
    "litellm>=1.50",
    "chromadb>=0.5",
    "fastembed>=0.4",
    "pydantic>=2.8",
    "pydantic-settings>=2.4",
    "sqlalchemy>=2.0",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "httpx>=0.27",
    "diskcache>=5.6",
    "structlog>=24.4",
    "rich>=13.7",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.3", "pytest-cov>=5.0", "respx>=0.21"]
evals = ["ragas>=0.2", "datasets>=3.0"]

[project.scripts]
tripmate = "tripmate.adapters.cli:main"
tripmate-ingest = "tripmate.rag.ingest:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tripmate"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
    "slow: requires a live LLM provider",
]

[tool.coverage.run]
source = ["src/tripmate"]
omit = ["*/adapters/cli.py"]
```

- [ ] **Step 4: Write `.gitignore` and install dependencies**

```bash
cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
.venv/
.env
.chroma/
.weather_cache/
*.db
.coverage
htmlcov/
traces/*.jsonl
!traces/example_*.jsonl
.pytest_cache/
EOF

uv venv
uv pip install -e ".[dev]"
```

- [ ] **Step 5: Write the failing test for configuration**

```python
# tests/test_config.py
import pytest

from tripmate.config import ConfigError, Settings


def test_provider_parsed_from_prefixed_model():
    settings = Settings(llm_model="anthropic/claude-haiku-4-5", llm_api_key="k")
    assert settings.provider == "anthropic"


def test_provider_defaults_to_openai_when_model_is_unprefixed():
    settings = Settings(llm_model="gpt-4o-mini", llm_api_key="k")
    assert settings.provider == "openai"


def test_export_provider_key_writes_provider_specific_env_var(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings(llm_model="groq/llama-3.3-70b-versatile", llm_api_key="secret")

    settings.export_provider_key()

    import os
    assert os.environ["GROQ_API_KEY"] == "secret"


def test_export_provider_key_is_a_noop_for_keyless_ollama(monkeypatch):
    settings = Settings(llm_model="ollama/qwen3.5", llm_api_key=None)

    settings.export_provider_key()  # must not raise


def test_export_provider_key_raises_when_required_key_is_missing():
    settings = Settings(llm_model="gpt-4o-mini", llm_api_key=None)

    with pytest.raises(ConfigError, match="LLM_API_KEY is required"):
        settings.export_provider_key()


def test_export_provider_key_raises_on_unknown_provider():
    settings = Settings(llm_model="madeup/model-x", llm_api_key="k")

    with pytest.raises(ConfigError, match="Unknown provider"):
        settings.export_provider_key()


def test_query_char_limit_must_be_positive():
    with pytest.raises(ValueError):
        Settings(llm_api_key="k", max_query_chars=0)
```

- [ ] **Step 6: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.config'`

- [ ] **Step 7: Implement `src/tripmate/config.py`**

```python
"""Application configuration. The single source of truth for every tunable."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
}

KEYLESS_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})

DEFAULT_PROVIDER = "openai"


class ConfigError(Exception):
    """Raised at startup when configuration is unusable."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM ---
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_timeout_s: int = Field(default=30, gt=0)

    # --- RAG ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chroma_path: str = "./.chroma"
    destinations_dir: str = "./data/destinations"
    rag_top_k: int = Field(default=4, gt=0)
    rag_min_score: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Agent ---
    max_tool_iterations: int = Field(default=5, gt=0)
    max_query_chars: int = Field(default=2000, gt=0)
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # --- Weather ---
    weather_timeout_s: float = Field(default=3.0, gt=0)
    weather_cache_dir: str = "./.weather_cache"
    weather_climate_years: int = Field(default=5, gt=0, le=20)

    # --- Persistence ---
    database_url: str = "sqlite:///./tripmate.db"

    # --- Observability ---
    log_level: str = "INFO"
    log_format: str = "rich"
    trace_dir: str = "./traces"

    @property
    def provider(self) -> str:
        """Provider slug taken from the model string, e.g. 'anthropic/claude' -> 'anthropic'."""
        if "/" not in self.llm_model:
            return DEFAULT_PROVIDER
        return self.llm_model.split("/", 1)[0]

    def export_provider_key(self) -> None:
        """Copy `llm_api_key` into the env var LiteLLM expects for this provider.

        LiteLLM reads provider-specific variables (OPENAI_API_KEY, GROQ_API_KEY, ...).
        Users configure one `LLM_API_KEY`; this maps it to the right place at startup.
        """
        provider = self.provider
        if provider in KEYLESS_PROVIDERS:
            return

        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var is None:
            supported = ", ".join(sorted(PROVIDER_ENV_VARS) + sorted(KEYLESS_PROVIDERS))
            raise ConfigError(
                f"Unknown provider {provider!r} in LLM_MODEL={self.llm_model!r}. "
                f"Supported: {supported}"
            )

        if not self.llm_api_key:
            raise ConfigError(
                f"LLM_API_KEY is required for provider {provider!r}. "
                f"Set it in .env, or switch LLM_MODEL to an ollama/* model."
            )

        os.environ[env_var] = self.llm_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call `get_settings.cache_clear()` in tests."""
    return Settings()
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 7 passed

- [ ] **Step 9: Write `.env.example`**

```bash
cat > .env.example <<'EOF'
# --- LLM (any provider; set exactly one key) ---
LLM_MODEL=gpt-4o-mini
# LLM_MODEL=anthropic/claude-haiku-4-5
# LLM_MODEL=groq/llama-3.3-70b-versatile
# LLM_MODEL=ollama/qwen3.5          # no key required
LLM_API_KEY=
LLM_BASE_URL=
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_S=30

# --- RAG ---
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_PATH=./.chroma
DESTINATIONS_DIR=./data/destinations
RAG_TOP_K=4
RAG_MIN_SCORE=0.25

# --- Agent ---
MAX_TOOL_ITERATIONS=5
MAX_QUERY_CHARS=2000
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_THRESHOLD=0.95

# --- Weather ---
WEATHER_TIMEOUT_S=3
WEATHER_CACHE_DIR=./.weather_cache
WEATHER_CLIMATE_YEARS=5

# --- Persistence ---
DATABASE_URL=sqlite:///./tripmate.db

# --- Observability ---
LOG_LEVEL=INFO
LOG_FORMAT=rich
TRACE_DIR=./traces
EOF
```

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/ tests/ data/
git commit -m "feat: project scaffold with validated provider-agnostic configuration"
```

---

### Task 2: Domain models

**Files:**
- Create: `src/tripmate/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces (all pydantic v2 `BaseModel` unless noted):
  - `Chunk(id: str, text: str, city: str, section: str, source_file: str, score: float = 0.0)`
  - `WeatherReport(city, period, source, conditions, temp_range_c: list[float], precip_days: int)`
  - `ToolStatus` — `Literal["ok", "no_data", "error"]`
  - `ToolResult(status: ToolStatus, data: dict | None, reason: str | None, tool_name: str)`; classmethods `ok(tool_name, data)`, `no_data(tool_name, reason, **extra)`, `error(tool_name, reason)`
  - `ToolCall(id: str, name: str, arguments: dict)`
  - `Citation(city: str, section: str)`; `.ref -> str` returns `"{city}/{SECTION}"`; classmethod `parse(text) -> Citation | None`
  - `LLMResponse(content: str | None, tool_calls: list[ToolCall], prompt_tokens: int, completion_tokens: int, cost_usd: float)`
  - `TraceEvent(seq: int, event_type: str, timestamp: float, duration_ms: float | None, payload: dict)`
  - `AgentResponse(answer, citations, trace, prompt_tokens, completion_tokens, cost_usd, latency_ms, session_id, was_cached)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from tripmate.models import Citation, Chunk, ToolResult, WeatherReport


def test_chunk_defaults_score_to_zero():
    chunk = Chunk(id="a1", text="t", city="tokyo", section="VISA & ENTRY",
                  source_file="tokyo.txt")
    assert chunk.score == 0.0


def test_citation_ref_is_city_slash_uppercase_section():
    assert Citation(city="tokyo", section="PACKING TIPS").ref == "tokyo/PACKING TIPS"


def test_citation_parse_accepts_bracketed_reference():
    citation = Citation.parse("[tokyo/PACKING TIPS]")
    assert citation == Citation(city="tokyo", section="PACKING TIPS")


def test_citation_parse_lowercases_city_and_uppercases_section():
    citation = Citation.parse("[Tokyo/packing tips]")
    assert citation.ref == "tokyo/PACKING TIPS"


def test_citation_parse_returns_none_for_malformed_text():
    assert Citation.parse("not a citation") is None


def test_tool_result_ok_carries_data_and_no_reason():
    result = ToolResult.ok("get_weather_forecast", {"temp": 5})
    assert result.status == "ok"
    assert result.data == {"temp": 5}
    assert result.reason is None


def test_tool_result_no_data_carries_reason_and_extra_fields():
    result = ToolResult.no_data(
        "search_destination_guide", "no guide for paris",
        available_cities=["tokyo"],
    )
    assert result.status == "no_data"
    assert result.reason == "no guide for paris"
    assert result.data == {"available_cities": ["tokyo"]}


def test_tool_result_error_has_error_status():
    result = ToolResult.error("get_weather_forecast", "timeout")
    assert result.status == "error"
    assert result.reason == "timeout"


def test_weather_report_serialises_to_dict():
    report = WeatherReport(
        city="Tokyo", period="December", source="climate_normal",
        conditions="cold, dry", temp_range_c=[3.0, 12.0], precip_days=4,
    )
    assert report.model_dump()["source"] == "climate_normal"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.models'`

- [ ] **Step 3: Implement `src/tripmate/models.py`**

```python
"""Shared domain models. Every boundary in the system speaks these types."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

CITATION_RE = re.compile(r"\[([A-Za-z\s\-']+)/([A-Za-z\s&'\-]+)\]")

ToolStatus = Literal["ok", "no_data", "error"]
WeatherSource = Literal["forecast", "climate_normal", "mock_fallback"]


class Chunk(BaseModel):
    """One retrievable section of a destination guide."""

    id: str
    text: str
    city: str
    section: str
    source_file: str
    score: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.city}/{self.section}"


class WeatherReport(BaseModel):
    city: str
    period: str
    source: WeatherSource
    conditions: str
    temp_range_c: list[float]
    precip_days: int


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Tools always return this. They never raise into the agent loop."""

    tool_name: str
    status: ToolStatus
    data: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def ok(cls, tool_name: str, data: dict[str, Any]) -> ToolResult:
        return cls(tool_name=tool_name, status="ok", data=data)

    @classmethod
    def no_data(cls, tool_name: str, reason: str, **extra: Any) -> ToolResult:
        return cls(
            tool_name=tool_name, status="no_data", reason=reason,
            data=dict(extra) if extra else None,
        )

    @classmethod
    def error(cls, tool_name: str, reason: str) -> ToolResult:
        return cls(tool_name=tool_name, status="error", reason=reason)


class Citation(BaseModel):
    city: str
    section: str

    @property
    def ref(self) -> str:
        return f"{self.city}/{self.section}"

    @classmethod
    def parse(cls, text: str) -> Citation | None:
        """Parse a single '[city/SECTION]' reference. Returns None when malformed."""
        match = CITATION_RE.search(text)
        if match is None:
            return None
        return cls(city=match.group(1).strip().lower(),
                   section=match.group(2).strip().upper())


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class TraceEvent(BaseModel):
    seq: int
    event_type: str
    timestamp: float
    duration_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    session_id: str = ""
    was_cached: bool = False
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/models.py tests/test_models.py
git commit -m "feat: domain models for chunks, tool results, citations and traces"
```

---

### Task 3: Tool registry

**Files:**
- Create: `src/tripmate/tools/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `ToolResult` from Task 2
- Produces:
  - `@tool` decorator — attaches `_tool_spec` to a function and returns it unchanged
  - `ToolSpec(name: str, description: str, fn: Callable, arg_model: type[BaseModel], schema: dict)`
  - `ToolRegistry`
    - `register(fn) -> None`
    - `schemas() -> list[dict]` — OpenAI `tools` array
    - `names() -> list[str]`
    - `dispatch(name: str, raw_args: dict) -> ToolResult` — validates args, executes, converts every exception to `ToolResult.error`

Argument descriptions come from `Annotated[type, "description"]`. There is no hand-written JSON schema anywhere in the codebase.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
from typing import Annotated

from tripmate.models import ToolResult
from tripmate.tools.registry import ToolRegistry, tool


@tool
def sample_add(
    first: Annotated[int, "The first number"],
    second: Annotated[int, "The second number"] = 10,
) -> ToolResult:
    """Add two numbers together."""
    return ToolResult.ok("sample_add", {"sum": first + second})


@tool
def sample_boom(city: Annotated[str, "A city"]) -> ToolResult:
    """Always explodes."""
    raise RuntimeError("kaboom")


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(sample_add)
    registry.register(sample_boom)
    return registry


def test_schema_uses_docstring_summary_as_description():
    schema = _registry().schemas()[0]
    assert schema["function"]["description"] == "Add two numbers together."


def test_schema_derives_parameter_types_from_type_hints():
    params = _registry().schemas()[0]["function"]["parameters"]
    assert params["properties"]["first"]["type"] == "integer"


def test_schema_carries_annotated_description():
    params = _registry().schemas()[0]["function"]["parameters"]
    assert params["properties"]["first"]["description"] == "The first number"


def test_schema_marks_only_defaultless_parameters_required():
    params = _registry().schemas()[0]["function"]["parameters"]
    assert params["required"] == ["first"]


def test_dispatch_executes_the_tool_and_returns_its_result():
    result = _registry().dispatch("sample_add", {"first": 1, "second": 2})
    assert result.data == {"sum": 3}


def test_dispatch_applies_declared_defaults():
    result = _registry().dispatch("sample_add", {"first": 5})
    assert result.data == {"sum": 15}


def test_dispatch_coerces_valid_string_input_to_declared_type():
    result = _registry().dispatch("sample_add", {"first": "7", "second": "3"})
    assert result.data == {"sum": 10}


def test_dispatch_returns_error_result_for_invalid_arguments():
    result = _registry().dispatch("sample_add", {"first": "not a number"})
    assert result.status == "error"
    assert "invalid arguments" in result.reason


def test_dispatch_returns_error_result_for_unknown_tool():
    result = _registry().dispatch("no_such_tool", {})
    assert result.status == "error"
    assert "unknown tool" in result.reason


def test_dispatch_converts_tool_exceptions_into_error_results():
    result = _registry().dispatch("sample_boom", {"city": "tokyo"})
    assert result.status == "error"
    assert "kaboom" in result.reason


def test_names_lists_every_registered_tool():
    assert _registry().names() == ["sample_add", "sample_boom"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.tools.registry'`

- [ ] **Step 3: Implement `src/tripmate/tools/registry.py`**

```python
"""Tool registry. Schemas are derived from type hints, never hand-written."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, ValidationError, create_model

from tripmate.models import ToolResult

TOOL_SPEC_ATTR = "_tool_spec"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., ToolResult]
    arg_model: type[BaseModel]
    schema: dict[str, Any]


def _describe(hint: Any) -> tuple[Any, str]:
    """Split `Annotated[T, "description"]` into (T, description)."""
    if get_origin(hint) is not Annotated:
        return hint, ""
    actual, *metadata = get_args(hint)
    description = next((m for m in metadata if isinstance(m, str)), "")
    return actual, description


def _build_arg_model(fn: Callable[..., Any]) -> type[BaseModel]:
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        annotation, description = _describe(hints.get(name, str))
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (annotation, Field(default, description=description))
    return create_model(f"{fn.__name__}_Args", **fields)


def _build_schema(fn: Callable[..., Any], arg_model: type[BaseModel]) -> dict[str, Any]:
    parameters = arg_model.model_json_schema()
    parameters.pop("title", None)
    for prop in parameters.get("properties", {}).values():
        prop.pop("title", None)
    description = (fn.__doc__ or "").strip().split("\n")[0]
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": parameters,
        },
    }


def tool(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Mark a function as an agent tool and attach its derived schema."""
    arg_model = _build_arg_model(fn)
    setattr(
        fn,
        TOOL_SPEC_ATTR,
        ToolSpec(
            name=fn.__name__,
            description=(fn.__doc__ or "").strip().split("\n")[0],
            fn=fn,
            arg_model=arg_model,
            schema=_build_schema(fn, arg_model),
        ),
    )
    return fn


class ToolRegistry:
    """Holds tool specs and dispatches calls with validated arguments."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, fn: Callable[..., ToolResult]) -> None:
        spec = getattr(fn, TOOL_SPEC_ATTR, None)
        if spec is None:
            raise ValueError(f"{fn.__name__} is not decorated with @tool")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._specs)

    def schemas(self) -> list[dict[str, Any]]:
        """Every tool schema, in registration order.

        Scale hook: past ~10 tools this is where semantic tool routing plugs in —
        embed the descriptions and return only the top-K relevant to the query.
        """
        return [spec.schema for spec in self._specs.values()]

    def dispatch(self, name: str, raw_args: dict[str, Any]) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult.error(name, f"unknown tool {name!r}")

        try:
            args = spec.arg_model(**raw_args)
        except ValidationError as exc:
            return ToolResult.error(name, f"invalid arguments: {exc.errors()}")

        try:
            return spec.fn(**args.model_dump())
        except Exception as exc:  # tools must never raise into the loop
            return ToolResult.error(name, f"{type(exc).__name__}: {exc}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/tools/registry.py tests/test_registry.py
git commit -m "feat: tool registry deriving JSON schemas from type hints"
```

---

### Task 4: Destination guide chunker

**Files:**
- Create: `src/tripmate/rag/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `Chunk` from Task 2
- Produces:
  - `parse_guide(path: Path) -> list[Chunk]` — one chunk per section of one city file
  - `load_all(directory: Path) -> list[Chunk]` — every `*.txt` in the directory, sorted by filename
  - `chunk_id(city: str, section: str, body: str) -> str` — 16-char sha256 prefix, stable across runs

Chunk text is `"{City} — {SECTION}\n{body}"`, so the city name is embedded as well as stored as metadata. Sections are already semantically self-contained, so no overlap is applied.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunker.py
from pathlib import Path

import pytest

from tripmate.rag.chunker import chunk_id, load_all, parse_guide

GUIDE = """DESTINATION GUIDE: TOKYO, JAPAN

Note: This is a simplified reference document.

VISA & ENTRY
Many nationalities can enter Japan visa-free for short tourist stays.

BEST TIME TO VISIT
Spring is popular for cherry blossoms.

PACKING TIPS
Layered clothing works well given seasonal variation.
"""


@pytest.fixture()
def guide_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokyo.txt"
    path.write_text(GUIDE, encoding="utf-8")
    return path


def test_parse_guide_returns_one_chunk_per_section(guide_file: Path):
    assert len(parse_guide(guide_file)) == 3


def test_parse_guide_extracts_city_from_the_title_line(guide_file: Path):
    assert {c.city for c in parse_guide(guide_file)} == {"tokyo"}


def test_parse_guide_preserves_section_headings(guide_file: Path):
    sections = [c.section for c in parse_guide(guide_file)]
    assert sections == ["VISA & ENTRY", "BEST TIME TO VISIT", "PACKING TIPS"]


def test_chunk_text_embeds_the_city_name_for_retrieval(guide_file: Path):
    first = parse_guide(guide_file)[0]
    assert first.text.startswith("Tokyo — VISA & ENTRY")


def test_parse_guide_excludes_the_disclaimer_note(guide_file: Path):
    assert all("simplified reference" not in c.text for c in parse_guide(guide_file))


def test_chunk_ids_are_stable_across_calls(guide_file: Path):
    assert [c.id for c in parse_guide(guide_file)] == [c.id for c in parse_guide(guide_file)]


def test_chunk_id_changes_when_body_changes():
    assert chunk_id("tokyo", "PACKING TIPS", "a") != chunk_id("tokyo", "PACKING TIPS", "b")


def test_load_all_reads_every_txt_file(tmp_path: Path):
    (tmp_path / "tokyo.txt").write_text(GUIDE, encoding="utf-8")
    (tmp_path / "osaka.txt").write_text(GUIDE.replace("TOKYO", "OSAKA"), encoding="utf-8")

    assert len(load_all(tmp_path)) == 6


def test_load_all_raises_when_directory_is_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path / "nope")


def test_real_data_pack_yields_twenty_chunks():
    chunks = load_all(Path("data/destinations"))
    assert len(chunks) == 20
    assert {c.city for c in chunks} == {"tokyo", "reykjavik", "bangkok", "barcelona"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.rag.chunker'`

- [ ] **Step 3: Implement `src/tripmate/rag/chunker.py`**

```python
"""Section-aware chunking of destination guides.

Each guide has a title line and five ALL-CAPS section headings. One section is one
chunk: they are already semantically self-contained, so no overlap is needed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tripmate.models import Chunk

SECTION_HEADING_RE = re.compile(r"^([A-Z][A-Z&\s]{3,})$")
TITLE_PREFIX = "DESTINATION GUIDE:"
ID_LENGTH = 16


def chunk_id(city: str, section: str, body: str) -> str:
    """Deterministic content hash. Re-ingesting unchanged content is a no-op."""
    digest = hashlib.sha256(f"{city}|{section}|{body}".encode("utf-8"))
    return digest.hexdigest()[:ID_LENGTH]


def _extract_city(lines: list[str]) -> str:
    for line in lines:
        if line.startswith(TITLE_PREFIX):
            location = line[len(TITLE_PREFIX):].strip()
            return location.split(",")[0].strip().lower()
    raise ValueError(f"no '{TITLE_PREFIX}' title line found")


def _split_sections(lines: list[str]) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        stripped = line.strip()
        if SECTION_HEADING_RE.match(stripped):
            sections.append((stripped, []))
        elif sections and stripped:
            sections[-1][1].append(stripped)
    return [(name, " ".join(body)) for name, body in sections if body]


def parse_guide(path: Path) -> list[Chunk]:
    """Parse one destination guide into one chunk per section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    city = _extract_city(lines)
    title = city.title()

    return [
        Chunk(
            id=chunk_id(city, section, body),
            text=f"{title} — {section}\n{body}",
            city=city,
            section=section,
            source_file=path.name,
        )
        for section, body in _split_sections(lines)
    ]


def load_all(directory: Path) -> list[Chunk]:
    """Parse every `*.txt` guide in `directory`, ordered by filename."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"destinations directory not found: {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.txt")):
        chunks.extend(parse_guide(path))
    return chunks
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: 10 passed

If `test_real_data_pack_yields_twenty_chunks` fails, print the parsed sections with
`python -c "from pathlib import Path; from tripmate.rag.chunker import load_all; [print(c.city, '|', c.section) for c in load_all(Path('data/destinations'))]"`
and adjust `SECTION_HEADING_RE` to match the actual headings.

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/rag/chunker.py tests/test_chunker.py
git commit -m "feat: section-aware guide chunker with content-hashed ids"
```

---

### Task 5: Vector store and ingest

**Files:**
- Create: `src/tripmate/rag/store.py`, `src/tripmate/rag/ingest.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `Chunk` (Task 2), `load_all` (Task 4), `get_settings` (Task 1)
- Produces:
  - `VectorStore` Protocol — `add(chunks) -> int`, `search(query, k, city, min_score) -> list[Chunk]`, `count() -> int`
  - `ChromaStore(path: str, embedding_model: str, collection: str = "destinations")` implementing it
  - `build_store(settings=None) -> ChromaStore` — factory reading configuration
  - `ingest(directory=None, settings=None) -> int` — returns the number of newly added chunks
  - `main() -> None` — console-script entry point for `tripmate-ingest`

Chroma returns **cosine distance**. `ChromaStore.search` converts with `similarity = 1 - distance` before comparing against `min_score`, so `min_score` is always a similarity in `[0, 1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from pathlib import Path

import pytest

from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore

CITIES = {"tokyo", "reykjavik", "bangkok", "barcelona"}


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> ChromaStore:
    path = tmp_path_factory.mktemp("chroma")
    store = ChromaStore(path=str(path), embedding_model="BAAI/bge-small-en-v1.5")
    store.add(load_all(Path("data/destinations")))
    return store


def test_store_holds_every_chunk_from_the_data_pack(store: ChromaStore):
    assert store.count() == 20


def test_re_adding_the_same_chunks_adds_nothing(store: ChromaStore):
    assert store.add(load_all(Path("data/destinations"))) == 0


def test_visa_query_retrieves_a_visa_section(store: ChromaStore):
    results = store.search("do I need a visa for Japan?", k=3, min_score=0.0)
    assert any("VISA" in chunk.section for chunk in results)


def test_city_filter_restricts_results_to_that_city(store: ChromaStore):
    results = store.search("what should I pack?", k=5, city="bangkok", min_score=0.0)
    assert {chunk.city for chunk in results} == {"bangkok"}


def test_city_filter_is_case_insensitive(store: ChromaStore):
    results = store.search("packing", k=5, city="Bangkok", min_score=0.0)
    assert {chunk.city for chunk in results} == {"bangkok"}


def test_scores_are_similarities_between_zero_and_one(store: ChromaStore):
    results = store.search("visa requirements", k=3, min_score=0.0)
    assert all(0.0 <= chunk.score <= 1.0 for chunk in results)


def test_results_are_ordered_by_descending_similarity(store: ChromaStore):
    scores = [c.score for c in store.search("local customs", k=4, min_score=0.0)]
    assert scores == sorted(scores, reverse=True)


def test_high_score_floor_filters_out_unrelated_queries(store: ChromaStore):
    assert store.search("how do I refinance a mortgage?", k=4, min_score=0.9) == []


def test_unknown_city_filter_returns_nothing(store: ChromaStore):
    assert store.search("visa", k=4, city="paris", min_score=0.0) == []


def test_top_k_limits_the_number_of_results(store: ChromaStore):
    assert len(store.search("travel", k=2, min_score=0.0)) <= 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.rag.store'`

- [ ] **Step 3: Implement `src/tripmate/rag/store.py`**

```python
"""Vector store. One Protocol, one implementation.

The Protocol is the swap point — Chroma to pgvector or Qdrant changes this file
and nothing else. A second implementation today would be speculative.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chromadb
from fastembed import TextEmbedding

from tripmate.config import Settings, get_settings
from tripmate.models import Chunk

COLLECTION_NAME = "destinations"
COSINE_SPACE = {"hnsw:space": "cosine"}


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> int: ...

    def search(
        self, query: str, k: int, city: str | None = None, min_score: float = 0.0
    ) -> list[Chunk]: ...

    def count(self) -> int: ...


class ChromaStore:
    """Persistent Chroma collection with local ONNX embeddings."""

    def __init__(
        self,
        path: str,
        embedding_model: str,
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata=COSINE_SPACE
        )
        self._embedder = TextEmbedding(model_name=embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._embedder.embed(texts)]

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk]) -> int:
        """Add only chunks whose content hash is not already stored. Returns how many."""
        if not chunks:
            return 0

        known = set(self._collection.get(ids=[c.id for c in chunks])["ids"])
        fresh = [c for c in chunks if c.id not in known]
        if not fresh:
            return 0

        self._collection.add(
            ids=[c.id for c in fresh],
            embeddings=self._embed([c.text for c in fresh]),
            documents=[c.text for c in fresh],
            metadatas=[
                {"city": c.city, "section": c.section, "source_file": c.source_file}
                for c in fresh
            ],
        )
        return len(fresh)

    def search(
        self, query: str, k: int, city: str | None = None, min_score: float = 0.0
    ) -> list[Chunk]:
        """Semantic search, optionally pre-filtered by city.

        The city filter is what keeps precision usable as the corpus grows: without it,
        20k chunks return another city's visa section for a Tokyo query.
        """
        if not query.strip():
            return []

        response = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=k,
            where={"city": city.strip().lower()} if city else None,
        )

        if not response["ids"] or not response["ids"][0]:
            return []

        results: list[Chunk] = []
        for chunk_id, document, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            similarity = 1.0 - float(distance)
            if similarity < min_score:
                continue
            results.append(
                Chunk(
                    id=chunk_id,
                    text=document,
                    city=str(metadata["city"]),
                    section=str(metadata["section"]),
                    source_file=str(metadata["source_file"]),
                    score=similarity,
                )
            )
        return sorted(results, key=lambda c: c.score, reverse=True)


def build_store(settings: Settings | None = None) -> ChromaStore:
    settings = settings or get_settings()
    return ChromaStore(
        path=settings.chroma_path, embedding_model=settings.embedding_model
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: 10 passed. The first run downloads the ~50MB embedding model.

- [ ] **Step 5: Implement `src/tripmate/rag/ingest.py`**

```python
"""Idempotent ingest of the destination data pack into the vector store."""

from __future__ import annotations

from pathlib import Path

from tripmate.config import Settings, get_settings
from tripmate.rag.chunker import load_all
from tripmate.rag.store import build_store


def ingest(directory: str | Path | None = None, settings: Settings | None = None) -> int:
    """Ingest every guide. Returns the number of chunks newly added."""
    settings = settings or get_settings()
    chunks = load_all(Path(directory or settings.destinations_dir))
    return build_store(settings).add(chunks)


def main() -> None:
    settings = get_settings()
    added = ingest(settings=settings)
    total = build_store(settings).count()
    print(f"ingested {added} new chunk(s); {total} total in {settings.chroma_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run ingest against the real data pack**

Run: `uv run tripmate-ingest`
Expected: `ingested 20 new chunk(s); 20 total in ./.chroma`

Run it a second time.
Expected: `ingested 0 new chunk(s); 20 total in ./.chroma` — proving idempotency.

- [ ] **Step 7: Commit**

```bash
git add src/tripmate/rag/store.py src/tripmate/rag/ingest.py tests/test_store.py
git commit -m "feat: chroma vector store with city filtering and idempotent ingest"
```

---

### Task 6: Destination guide tool

**Files:**
- Create: `src/tripmate/tools/destination.py`
- Test: `tests/test_tools_destination.py`

**Interfaces:**
- Consumes: `@tool` (Task 3), `build_store` (Task 5), `ToolResult` (Task 2)
- Produces:
  - `search_destination_guide(query: str, city: str | None = None) -> ToolResult` — decorated with `@tool`
  - `set_store(store) -> None` / `reset_store() -> None` — test seams for dependency substitution
  - `SUPPORTED_CITIES: tuple[str, ...]`

On an empty result the tool returns `ToolResult.no_data(...)` carrying `available_cities`, so the agent can name what it does cover instead of inventing content.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_destination.py
from pathlib import Path

import pytest

from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import (
    SUPPORTED_CITIES,
    reset_store,
    search_destination_guide,
    set_store,
)


@pytest.fixture(autouse=True, scope="module")
def _store(tmp_path_factory):
    store = ChromaStore(
        path=str(tmp_path_factory.mktemp("chroma_tool")),
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    store.add(load_all(Path("data/destinations")))
    set_store(store)
    yield
    reset_store()


def test_returns_ok_status_for_a_covered_topic():
    assert search_destination_guide("visa requirements for Japan").status == "ok"


def test_result_data_contains_chunk_dicts_with_refs():
    result = search_destination_guide("visa requirements for Japan")
    assert "ref" in result.data["chunks"][0]


def test_city_argument_restricts_results():
    result = search_destination_guide("what should I pack?", city="reykjavik")
    assert {c["city"] for c in result.data["chunks"]} == {"reykjavik"}


def test_unknown_city_returns_no_data_with_supported_cities():
    result = search_destination_guide("visa rules", city="paris")
    assert result.status == "no_data"
    assert set(result.data["available_cities"]) == set(SUPPORTED_CITIES)


def test_unknown_city_reason_names_the_city():
    assert "paris" in search_destination_guide("visa", city="paris").reason


def test_empty_query_returns_no_data():
    assert search_destination_guide("   ").status == "no_data"


def test_tool_carries_a_derived_schema():
    schema = search_destination_guide._tool_spec.schema
    assert schema["function"]["name"] == "search_destination_guide"
    assert "city" in schema["function"]["parameters"]["properties"]


def test_city_is_optional_in_the_schema():
    schema = search_destination_guide._tool_spec.schema
    assert "city" not in schema["function"]["parameters"].get("required", [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_tools_destination.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.tools.destination'`

- [ ] **Step 3: Implement `src/tripmate/tools/destination.py`**

```python
"""RAG tool over the destination knowledge base."""

from __future__ import annotations

from typing import Annotated

from tripmate.config import get_settings
from tripmate.models import ToolResult
from tripmate.rag.store import VectorStore, build_store
from tripmate.tools.registry import tool

SUPPORTED_CITIES: tuple[str, ...] = ("tokyo", "reykjavik", "bangkok", "barcelona")
TOOL_NAME = "search_destination_guide"

_store: VectorStore | None = None


def set_store(store: VectorStore) -> None:
    """Inject a store. Used by tests and by application startup."""
    global _store
    _store = store


def reset_store() -> None:
    global _store
    _store = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = build_store()
    return _store


@tool
def search_destination_guide(
    query: Annotated[
        str,
        "The travel question or topic, e.g. 'visa requirements' or 'what to pack'.",
    ],
    city: Annotated[
        str | None,
        "Destination city to restrict the search to. One of: tokyo, reykjavik, "
        "bangkok, barcelona. Omit when the question is not about a specific city.",
    ] = None,
) -> ToolResult:
    """Search the destination knowledge base for visa, weather-season, customs, packing and safety guidance."""
    if not query or not query.strip():
        return ToolResult.no_data(TOOL_NAME, "empty query")

    settings = get_settings()
    chunks = _get_store().search(
        query=query,
        k=settings.rag_top_k,
        city=city,
        min_score=settings.rag_min_score,
    )

    if not chunks:
        target = city.strip().lower() if city else "that topic"
        return ToolResult.no_data(
            TOOL_NAME,
            f"no destination guide content found for {target}",
            available_cities=list(SUPPORTED_CITIES),
        )

    return ToolResult.ok(
        TOOL_NAME,
        {
            "chunks": [
                {
                    "ref": chunk.ref,
                    "city": chunk.city,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": round(chunk.score, 4),
                }
                for chunk in chunks
            ]
        },
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_tools_destination.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/tools/destination.py tests/test_tools_destination.py
git commit -m "feat: destination guide RAG tool with explicit no-data handling"
```

---

### Task 7: Weather forecast tool

**Files:**
- Create: `src/tripmate/tools/weather.py`
- Test: `tests/test_tools_weather.py`

**Interfaces:**
- Consumes: `@tool` (Task 3), `ToolResult`, `WeatherReport` (Task 2), settings (Task 1)
- Produces:
  - `get_weather_forecast(city: str, date_or_month: str) -> ToolResult` — decorated with `@tool`
  - `resolve_period(text: str, today: date | None = None) -> Period` where `Period(kind: Literal["date","month"], date: date | None, month: int, label: str)`
  - `describe(temp_min, temp_max, precip_days) -> str` — conditions phrase
  - `MOCK_CLIMATE: dict[str, dict[str, tuple[float, float, int, str]]]`
  - `clear_cache() -> None`

Routing: an ISO date within the forecast horizon uses the forecast endpoint; anything else (a month name, or a date beyond the horizon) uses the archive endpoint averaged over the last N years — a **climate normal**, not a forecast. The `source` field always states which path produced the answer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_weather.py
from datetime import date

import httpx
import pytest
import respx

from tripmate.tools.weather import (
    ARCHIVE_URL,
    FORECAST_URL,
    GEOCODE_URL,
    clear_cache,
    describe,
    get_weather_forecast,
    resolve_period,
)

GEOCODE_OK = {"results": [{"latitude": 35.68, "longitude": 139.75, "name": "Tokyo"}]}


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


def test_resolve_period_parses_a_month_name():
    period = resolve_period("December")
    assert period.kind == "month"
    assert period.month == 12


def test_resolve_period_parses_an_abbreviated_month():
    assert resolve_period("dec").month == 12


def test_resolve_period_parses_an_iso_date_near_today():
    today = date(2026, 9, 16)
    period = resolve_period("2026-09-20", today=today)
    assert period.kind == "date"


def test_resolve_period_downgrades_a_far_future_date_to_its_month():
    today = date(2026, 9, 16)
    period = resolve_period("2026-12-25", today=today)
    assert period.kind == "month"
    assert period.month == 12


def test_resolve_period_rejects_unparseable_text():
    with pytest.raises(ValueError):
        resolve_period("sometime soonish")


def test_describe_reports_cold_for_low_temperatures():
    assert "cold" in describe(-3.0, 2.0, 2)


def test_describe_reports_hot_for_high_temperatures():
    assert "hot" in describe(30.0, 36.0, 1)


def test_describe_mentions_rain_when_precipitation_days_are_high():
    assert "rain" in describe(20.0, 26.0, 18)


@respx.mock
def test_month_query_uses_the_archive_endpoint_and_labels_climate_normal():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2021-12-01", "2021-12-02", "2022-12-01"],
                "temperature_2m_max": [12.0, 11.0, 13.0],
                "temperature_2m_min": [3.0, 2.0, 4.0],
                "precipitation_sum": [0.0, 5.0, 0.0],
            }
        })
    )

    result = get_weather_forecast("Tokyo", "December")

    assert result.status == "ok"
    assert result.data["source"] == "climate_normal"
    assert result.data["period"] == "December"


@respx.mock
def test_near_date_query_uses_the_forecast_endpoint():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))
    respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2026-09-18"],
                "temperature_2m_max": [26.0],
                "temperature_2m_min": [19.0],
                "precipitation_sum": [0.0],
            }
        })
    )

    result = get_weather_forecast("Tokyo", date.today().isoformat())

    assert result.data["source"] == "forecast"


@respx.mock
def test_unknown_city_returns_no_data():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    result = get_weather_forecast("Atlantis", "July")

    assert result.status == "no_data"
    assert "Atlantis" in result.reason


@respx.mock
def test_api_timeout_falls_back_to_the_mock_table_for_covered_cities():
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    result = get_weather_forecast("Tokyo", "December")

    assert result.status == "ok"
    assert result.data["source"] == "mock_fallback"


@respx.mock
def test_api_failure_for_an_uncovered_city_returns_error():
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    result = get_weather_forecast("Lisbon", "December")

    assert result.status == "error"


@respx.mock
def test_malformed_period_returns_no_data_without_calling_the_api():
    route = respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))

    result = get_weather_forecast("Tokyo", "whenever")

    assert result.status == "no_data"
    assert not route.called


@respx.mock
def test_repeated_month_query_is_served_from_cache():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_OK))
    archive = respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2021-12-01"],
                "temperature_2m_max": [12.0],
                "temperature_2m_min": [3.0],
                "precipitation_sum": [0.0],
            }
        })
    )

    get_weather_forecast("Tokyo", "December")
    get_weather_forecast("Tokyo", "December")

    assert archive.call_count == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_tools_weather.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.tools.weather'`

- [ ] **Step 3: Implement `src/tripmate/tools/weather.py`**

```python
"""Weather tool over Open-Meteo, with a climate-normal path and a mock fallback.

A forecast API reaches about 16 days. "What should I pack for Tokyo in December?"
is not a forecast question — it is a climate-normal question, answered from
historical reanalysis averaged over the last N years. The `source` field on every
response states which path produced it, so the agent never misrepresents what it knows.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean
from typing import Annotated, Literal

import diskcache
import httpx

from tripmate.config import get_settings
from tripmate.models import ToolResult, WeatherReport
from tripmate.tools.registry import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HORIZON_DAYS = 16
ARCHIVE_LAG_DAYS = 7
DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
RAINY_DAY_MM = 1.0
FORECAST_TTL_S = 3600

COLD_MAX_C = 10.0
HOT_MIN_C = 28.0
WET_RATIO = 0.4
DRY_RATIO = 0.15

TOOL_NAME = "get_weather_forecast"

MONTH_NUMBERS: dict[str, int] = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name)
    if name
} | {
    name.lower(): index
    for index, name in enumerate(calendar.month_abbr)
    if name
}

# Seasonal fallback used only when Open-Meteo is unreachable.
# (temp_min_c, temp_max_c, precip_days, conditions)
MOCK_CLIMATE: dict[str, dict[str, tuple[float, float, int, str]]] = {
    "tokyo": {
        "winter": (2.0, 10.0, 5, "cold, dry, mostly clear"),
        "spring": (10.0, 20.0, 10, "mild, occasional showers"),
        "summer": (23.0, 31.0, 14, "hot, humid, frequent rain"),
        "autumn": (14.0, 23.0, 11, "mild, some rain"),
    },
    "reykjavik": {
        "winter": (-3.0, 3.0, 15, "cold, windy, occasional snow"),
        "spring": (1.0, 8.0, 12, "chilly, changeable, windy"),
        "summer": (8.0, 14.0, 11, "cool, breezy, long daylight"),
        "autumn": (2.0, 8.0, 15, "cold, wet, windy"),
    },
    "bangkok": {
        "winter": (21.0, 32.0, 2, "hot, dry, sunny"),
        "spring": (26.0, 35.0, 6, "very hot, humid"),
        "summer": (25.0, 33.0, 17, "hot, humid, heavy monsoon rain"),
        "autumn": (24.0, 32.0, 18, "hot, humid, frequent heavy rain"),
    },
    "barcelona": {
        "winter": (5.0, 14.0, 5, "mild, some rain"),
        "spring": (10.0, 20.0, 6, "mild, pleasant"),
        "summer": (20.0, 29.0, 3, "hot, dry, sunny"),
        "autumn": (13.0, 23.0, 8, "mild, occasional heavy rain"),
    },
}

SEASONS: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(get_settings().weather_cache_dir)
    return _cache


def clear_cache() -> None:
    _get_cache().clear()


@dataclass(frozen=True)
class Period:
    kind: Literal["date", "month"]
    month: int
    label: str
    day: date | None = None


def resolve_period(text: str, today: date | None = None) -> Period:
    """Classify a date-or-month string.

    An ISO date inside the forecast horizon stays a date; anything else becomes a
    month, because only a climate normal can answer it.
    """
    today = today or date.today()
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty date_or_month")

    try:
        parsed = datetime.strptime(cleaned, "%Y-%m-%d").date()
    except ValueError:
        month = MONTH_NUMBERS.get(cleaned.lower())
        if month is None:
            raise ValueError(f"cannot interpret {text!r} as a date or month")
        return Period(kind="month", month=month, label=calendar.month_name[month])

    delta_days = (parsed - today).days
    if 0 <= delta_days <= FORECAST_HORIZON_DAYS:
        return Period(kind="date", month=parsed.month, label=parsed.isoformat(),
                      day=parsed)
    return Period(kind="month", month=parsed.month,
                  label=calendar.month_name[parsed.month])


def describe(temp_min: float, temp_max: float, precip_days: int,
             total_days: int = 30) -> str:
    """Turn numbers into a short human phrase."""
    if temp_max <= COLD_MAX_C:
        temperature = "cold"
    elif temp_min >= HOT_MIN_C:
        temperature = "hot"
    elif temp_max >= 24.0:
        temperature = "warm"
    else:
        temperature = "mild"

    ratio = precip_days / max(total_days, 1)
    if ratio >= WET_RATIO:
        precipitation = "frequent rain"
    elif ratio <= DRY_RATIO:
        precipitation = "mostly dry"
    else:
        precipitation = "occasional rain"

    return f"{temperature}, {precipitation}"


def _geocode(city: str, timeout: float) -> tuple[float, float, str] | None:
    response = httpx.get(
        GEOCODE_URL, params={"name": city, "count": 1}, timeout=timeout
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    first = results[0]
    return float(first["latitude"]), float(first["longitude"]), str(first["name"])


def _summarise(daily: dict, month: int | None) -> tuple[float, float, int, int]:
    """Average max/min temperature and count rainy days, optionally for one month."""
    times = daily["time"]
    maxes, mins, precip = [], [], []
    for index, stamp in enumerate(times):
        if month is not None and int(stamp.split("-")[1]) != month:
            continue
        maxes.append(daily["temperature_2m_max"][index])
        mins.append(daily["temperature_2m_min"][index])
        precip.append(daily["precipitation_sum"][index] or 0.0)

    if not maxes:
        raise ValueError("no daily records for the requested period")

    rainy = sum(1 for value in precip if value >= RAINY_DAY_MM)
    days = len(maxes)
    normalised_rainy = round(rainy * 30 / days) if days else 0
    return mean(mins), mean(maxes), normalised_rainy, days


def _fetch_forecast(lat: float, lon: float, day: date, timeout: float) -> dict:
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": lat, "longitude": lon, "daily": DAILY_FIELDS,
            "start_date": day.isoformat(), "end_date": day.isoformat(),
            "timezone": "auto",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["daily"]


def _fetch_archive(lat: float, lon: float, years: int, timeout: float) -> dict:
    end = date.today().replace(day=1)
    start = end.replace(year=end.year - years)
    response = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon, "daily": DAILY_FIELDS,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "auto",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["daily"]


def _mock_report(city: str, period: Period) -> WeatherReport | None:
    seasons = MOCK_CLIMATE.get(city.strip().lower())
    if seasons is None:
        return None
    low, high, rainy, conditions = seasons[SEASONS[period.month]]
    return WeatherReport(
        city=city.title(), period=period.label, source="mock_fallback",
        conditions=conditions, temp_range_c=[low, high], precip_days=rainy,
    )


@tool
def get_weather_forecast(
    city: Annotated[str, "City name, e.g. 'Tokyo'."],
    date_or_month: Annotated[
        str,
        "An ISO date like '2026-12-25' for near-term forecasts, or a month name "
        "like 'December' for typical seasonal conditions.",
    ],
) -> ToolResult:
    """Get the expected weather for a city on a specific date or during a given month."""
    settings = get_settings()

    try:
        period = resolve_period(date_or_month)
    except ValueError as exc:
        return ToolResult.no_data(TOOL_NAME, str(exc))

    cache_key = f"{period.kind}:{city.strip().lower()}:{period.label}"
    cached = _get_cache().get(cache_key)
    if cached is not None:
        return ToolResult.ok(TOOL_NAME, cached)

    try:
        located = _geocode(city, settings.weather_timeout_s)
        if located is None:
            return ToolResult.no_data(
                TOOL_NAME, f"could not find a city named {city!r}"
            )
        latitude, longitude, resolved_name = located

        if period.kind == "date" and period.day is not None:
            daily = _fetch_forecast(latitude, longitude, period.day,
                                    settings.weather_timeout_s)
            low, high, rainy, days = _summarise(daily, month=None)
            source = "forecast"
        else:
            daily = _fetch_archive(latitude, longitude,
                                   settings.weather_climate_years,
                                   settings.weather_timeout_s)
            low, high, rainy, days = _summarise(daily, month=period.month)
            source = "climate_normal"

        report = WeatherReport(
            city=resolved_name, period=period.label, source=source,
            conditions=describe(low, high, rainy, total_days=min(days, 30)),
            temp_range_c=[round(low, 1), round(high, 1)], precip_days=rainy,
        )

    except (httpx.HTTPError, KeyError, ValueError) as exc:
        fallback = _mock_report(city, period)
        if fallback is None:
            return ToolResult.error(
                TOOL_NAME,
                f"weather lookup failed for {city!r} and no offline data is "
                f"available: {type(exc).__name__}: {exc}",
            )
        payload = fallback.model_dump()
        return ToolResult.ok(TOOL_NAME, payload)

    payload = report.model_dump()
    if source == "climate_normal":
        _get_cache().set(cache_key, payload)  # normals never change
    else:
        _get_cache().set(cache_key, payload, expire=FORECAST_TTL_S)
    return ToolResult.ok(TOOL_NAME, payload)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_tools_weather.py -v`
Expected: 15 passed

- [ ] **Step 5: Smoke-test against the live API**

```bash
uv run python -c "
from tripmate.tools.weather import get_weather_forecast, clear_cache
clear_cache()
print(get_weather_forecast('Tokyo', 'December').data)
print(get_weather_forecast('Reykjavik', 'January').data)
"
```

Expected: two dicts with `source: climate_normal` and plausible temperature ranges — Tokyo mild-cold, Reykjavik near freezing.

- [ ] **Step 6: Commit**

```bash
git add src/tripmate/tools/weather.py tests/test_tools_weather.py
git commit -m "feat: weather tool with forecast, climate-normal and fallback paths"
```

---

### Task 8: LLM client and test double

**Files:**
- Create: `src/tripmate/llm/client.py`, `tests/fakes.py`
- Test: `tests/test_llm_client.py`

**Interfaces:**
- Consumes: `LLMResponse`, `ToolCall` (Task 2), settings (Task 1)
- Produces:
  - `LLMClient(model, temperature, timeout_s, base_url=None)` with `complete(messages: list[dict], tools: list[dict] | None = None) -> LLMResponse`
  - `build_llm_client(settings=None) -> LLMClient`
  - `LLMError(Exception)`
  - `FakeLLMClient(script: list[LLMResponse])` in `tests/fakes.py`, with `.calls: list[dict]` recording every request

`FakeLLMClient` is what makes tool-selection and integration tests deterministic: no API key, no network, no cost, no flake.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_client.py
import pytest

from tripmate.llm.client import LLMClient, LLMError
from tripmate.models import LLMResponse, ToolCall

from tests.fakes import FakeLLMClient


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FnCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _RawToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FnCall(name, arguments)


class _Choice:
    def __init__(self, message):
        self.message = message


class _Usage:
    prompt_tokens = 120
    completion_tokens = 40


class _Raw:
    def __init__(self, message):
        self.choices = [_Choice(message)]
        self.usage = _Usage()


def test_parses_plain_text_response():
    raw = _Raw(_Message(content="Hello there"))
    parsed = LLMClient._parse(raw)
    assert parsed.content == "Hello there"
    assert parsed.tool_calls == []


def test_parses_token_usage():
    parsed = LLMClient._parse(_Raw(_Message(content="hi")))
    assert parsed.prompt_tokens == 120
    assert parsed.completion_tokens == 40


def test_parses_tool_calls_with_json_arguments():
    raw = _Raw(_Message(tool_calls=[
        _RawToolCall("call_1", "get_weather_forecast",
                     '{"city": "Tokyo", "date_or_month": "December"}')
    ]))
    parsed = LLMClient._parse(raw)
    assert parsed.tool_calls[0].name == "get_weather_forecast"
    assert parsed.tool_calls[0].arguments["city"] == "Tokyo"


def test_malformed_tool_arguments_become_an_empty_dict():
    raw = _Raw(_Message(tool_calls=[_RawToolCall("c1", "t", "{not json")]))
    assert LLMClient._parse(raw).tool_calls[0].arguments == {}


def test_fake_client_replays_scripted_responses_in_order():
    fake = FakeLLMClient([
        LLMResponse(tool_calls=[ToolCall(id="1", name="t", arguments={})]),
        LLMResponse(content="done"),
    ])

    assert fake.complete([{"role": "user", "content": "q"}]).tool_calls[0].name == "t"
    assert fake.complete([{"role": "user", "content": "q"}]).content == "done"


def test_fake_client_records_every_request():
    fake = FakeLLMClient([LLMResponse(content="ok")])
    fake.complete([{"role": "user", "content": "q"}], tools=[{"x": 1}])

    assert fake.calls[0]["tools"] == [{"x": 1}]


def test_fake_client_raises_when_the_script_is_exhausted():
    fake = FakeLLMClient([LLMResponse(content="ok")])
    fake.complete([])

    with pytest.raises(LLMError, match="script exhausted"):
        fake.complete([])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.llm.client'`

- [ ] **Step 3: Implement `src/tripmate/llm/client.py`**

```python
"""Provider-agnostic LLM client.

LiteLLM normalises tool-calling across OpenAI, Anthropic, Gemini, Groq, Ollama and
others, so the agent loop is identical whichever model the user configures.
"""

from __future__ import annotations

import json
from typing import Any

import litellm

from tripmate.config import Settings, get_settings
from tripmate.models import LLMResponse, ToolCall


class LLMError(Exception):
    """Raised when the provider call fails after retries."""


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        timeout_s: int,
        base_url: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._timeout_s = timeout_s
        self._base_url = base_url
        self._max_retries = max_retries

    @staticmethod
    def _parse(raw: Any) -> LLMResponse:
        message = raw.choices[0].message
        usage = getattr(raw, "usage", None)

        tool_calls: list[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "timeout": self._timeout_s,
            "num_retries": self._max_retries,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self._base_url:
            kwargs["api_base"] = self._base_url

        try:
            raw = litellm.completion(**kwargs)
        except Exception as exc:
            raise LLMError(f"{self._model} call failed: {type(exc).__name__}: {exc}") from exc

        parsed = self._parse(raw)
        try:
            cost = float(litellm.completion_cost(completion_response=raw))
        except Exception:
            cost = 0.0  # unpriced or local model; not an error
        return parsed.model_copy(update={"cost_usd": cost})


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    settings.export_provider_key()
    return LLMClient(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
        base_url=settings.llm_base_url,
    )
```

- [ ] **Step 4: Implement `tests/fakes.py`**

```python
"""Test doubles. FakeLLMClient makes agent tests deterministic and free."""

from __future__ import annotations

from typing import Any

from tripmate.llm.client import LLMError
from tripmate.models import LLMResponse


class FakeLLMClient:
    """Replays a scripted list of LLMResponse objects in order."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages), "tools": tools})
        if self._index >= len(self._script):
            raise LLMError(
                f"script exhausted after {len(self._script)} response(s); "
                f"the agent made an unexpected extra call"
            )
        response = self._script[self._index]
        self._index += 1
        return response
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/tripmate/llm/client.py tests/fakes.py tests/test_llm_client.py
git commit -m "feat: litellm client with cost extraction and scripted test double"
```

---

### Task 9: Trace recorder

**Files:**
- Create: `src/tripmate/core/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Consumes: `TraceEvent` (Task 2)
- Produces:
  - `EventType` constants: `QUERY_RECEIVED`, `CACHE_HIT`, `LLM_CALL`, `TOOL_CALL`, `TOOL_RESULT`, `TOOL_ERROR`, `FALLBACK_USED`, `CITATION_REJECTED`, `ANSWER_SYNTHESIZED`
  - `Tracer(session_id: str, trace_dir: str | None = None)`
    - `record(event_type: str, duration_ms: float | None = None, **payload) -> TraceEvent`
    - `events: list[TraceEvent]`
    - `totals() -> TraceTotals` — `prompt_tokens`, `completion_tokens`, `cost_usd`, `latency_ms`, `tool_calls`, `llm_calls`
    - `flush() -> Path | None` — appends every event as JSONL to `{trace_dir}/{session_id}.jsonl`
  - `load_trace(path) -> list[TraceEvent]` — replay support

Traces are append-only and replayable: the log is the record of what happened, not a formatted string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace.py
import json

from tripmate.core.trace import EventType, Tracer, load_trace


def test_events_are_numbered_from_one():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.QUERY_RECEIVED, query="hi")
    tracer.record(EventType.LLM_CALL)

    assert [event.seq for event in tracer.events] == [1, 2]


def test_payload_keywords_are_stored_on_the_event():
    tracer = Tracer(session_id="s1")
    event = tracer.record(EventType.TOOL_CALL, tool="get_weather_forecast")

    assert event.payload["tool"] == "get_weather_forecast"


def test_totals_sum_tokens_and_cost_across_llm_calls():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.LLM_CALL, prompt_tokens=100, completion_tokens=20,
                  cost_usd=0.001)
    tracer.record(EventType.LLM_CALL, prompt_tokens=50, completion_tokens=10,
                  cost_usd=0.0005)

    totals = tracer.totals()
    assert totals.prompt_tokens == 150
    assert totals.completion_tokens == 30
    assert round(totals.cost_usd, 5) == 0.0015


def test_totals_count_llm_and_tool_calls_separately():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.LLM_CALL)
    tracer.record(EventType.TOOL_CALL, tool="a")
    tracer.record(EventType.TOOL_CALL, tool="b")

    totals = tracer.totals()
    assert totals.llm_calls == 1
    assert totals.tool_calls == 2


def test_totals_latency_sums_recorded_durations():
    tracer = Tracer(session_id="s1")
    tracer.record(EventType.LLM_CALL, duration_ms=120.0)
    tracer.record(EventType.TOOL_CALL, duration_ms=80.0)

    assert tracer.totals().latency_ms == 200.0


def test_flush_writes_one_json_object_per_line(tmp_path):
    tracer = Tracer(session_id="s1", trace_dir=str(tmp_path))
    tracer.record(EventType.QUERY_RECEIVED, query="hi")
    tracer.record(EventType.ANSWER_SYNTHESIZED, answer="hello")

    path = tracer.flush()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == EventType.QUERY_RECEIVED


def test_flush_appends_rather_than_overwriting(tmp_path):
    first = Tracer(session_id="s1", trace_dir=str(tmp_path))
    first.record(EventType.QUERY_RECEIVED, query="one")
    first.flush()

    second = Tracer(session_id="s1", trace_dir=str(tmp_path))
    second.record(EventType.QUERY_RECEIVED, query="two")
    path = second.flush()

    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_flush_is_a_noop_without_a_trace_directory():
    assert Tracer(session_id="s1").flush() is None


def test_load_trace_round_trips_written_events(tmp_path):
    tracer = Tracer(session_id="s1", trace_dir=str(tmp_path))
    tracer.record(EventType.TOOL_RESULT, tool="x", status="ok")
    path = tracer.flush()

    replayed = load_trace(path)
    assert replayed[0].payload["status"] == "ok"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.core.trace'`

- [ ] **Step 3: Implement `src/tripmate/core/trace.py`**

```python
"""Structured, append-only, replayable reasoning trace."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tripmate.models import TraceEvent


class EventType:
    QUERY_RECEIVED = "QUERY_RECEIVED"
    CACHE_HIT = "CACHE_HIT"
    LLM_CALL = "LLM_CALL"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    FALLBACK_USED = "FALLBACK_USED"
    CITATION_REJECTED = "CITATION_REJECTED"
    ANSWER_SYNTHESIZED = "ANSWER_SYNTHESIZED"


@dataclass(frozen=True)
class TraceTotals:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    llm_calls: int
    tool_calls: int


class Tracer:
    """Collects trace events for one session and writes them as JSONL."""

    def __init__(self, session_id: str, trace_dir: str | None = None) -> None:
        self.session_id = session_id
        self._trace_dir = trace_dir
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def record(
        self, event_type: str, duration_ms: float | None = None, **payload: Any
    ) -> TraceEvent:
        event = TraceEvent(
            seq=len(self._events) + 1,
            event_type=event_type,
            timestamp=time.time(),
            duration_ms=duration_ms,
            payload=payload,
        )
        self._events.append(event)
        return event

    def totals(self) -> TraceTotals:
        return TraceTotals(
            prompt_tokens=sum(e.payload.get("prompt_tokens", 0) for e in self._events),
            completion_tokens=sum(
                e.payload.get("completion_tokens", 0) for e in self._events
            ),
            cost_usd=sum(e.payload.get("cost_usd", 0.0) for e in self._events),
            latency_ms=sum(e.duration_ms or 0.0 for e in self._events),
            llm_calls=sum(1 for e in self._events if e.event_type == EventType.LLM_CALL),
            tool_calls=sum(1 for e in self._events if e.event_type == EventType.TOOL_CALL),
        )

    def flush(self) -> Path | None:
        """Append every event to `{trace_dir}/{session_id}.jsonl`."""
        if not self._trace_dir:
            return None

        directory = Path(self._trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.session_id}.jsonl"

        with path.open("a", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event.model_dump(), default=str) + "\n")
        return path


def load_trace(path: str | Path) -> list[TraceEvent]:
    """Read a JSONL trace back into events, for offline replay and auditing."""
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [TraceEvent(**json.loads(line)) for line in lines if line]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_trace.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/core/trace.py tests/test_trace.py
git commit -m "feat: append-only replayable trace recorder with cost accounting"
```

---

### Task 10: Persistence

**Files:**
- Create: `src/tripmate/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: settings (Task 1), `TraceEvent` (Task 2)
- Produces:
  - ORM models `SessionRow`, `TurnRow`, `TraceRow`, `CacheRow` on `Base`
  - `Database(url: str)` with `create_all()`, `session()` contextmanager
  - `build_database(settings=None) -> Database`
  - `SessionStore(db: Database)`
    - `ensure_session(session_id: str) -> None`
    - `add_turn(session_id, role, content, prompt_tokens=0, completion_tokens=0, cost_usd=0.0, latency_ms=0.0) -> int`
    - `add_trace(turn_id: int, events: list[TraceEvent]) -> None`
    - `history(session_id: str, limit: int = 20) -> list[dict]` — oldest first, shaped as LLM messages

SQLite by default; `DATABASE_URL=postgresql+psycopg://...` switches to PostgreSQL with no code change. Pooling is configured explicitly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import pytest

from tripmate.db import Database, SessionStore
from tripmate.models import TraceEvent


@pytest.fixture()
def store(tmp_path) -> SessionStore:
    db = Database(url=f"sqlite:///{tmp_path}/test.db")
    db.create_all()
    return SessionStore(db)


def test_history_is_empty_for_a_new_session(store: SessionStore):
    assert store.history("s1") == []


def test_add_turn_returns_an_increasing_id(store: SessionStore):
    first = store.add_turn("s1", "user", "hello")
    second = store.add_turn("s1", "assistant", "hi")
    assert second > first


def test_history_returns_messages_oldest_first(store: SessionStore):
    store.add_turn("s1", "user", "one")
    store.add_turn("s1", "assistant", "two")

    assert [m["content"] for m in store.history("s1")] == ["one", "two"]


def test_history_uses_llm_message_shape(store: SessionStore):
    store.add_turn("s1", "user", "hello")
    assert store.history("s1")[0] == {"role": "user", "content": "hello"}


def test_history_is_scoped_to_one_session(store: SessionStore):
    store.add_turn("s1", "user", "one")
    store.add_turn("s2", "user", "two")

    assert len(store.history("s1")) == 1


def test_history_limit_keeps_the_most_recent_turns(store: SessionStore):
    for index in range(5):
        store.add_turn("s1", "user", f"m{index}")

    recent = store.history("s1", limit=2)
    assert [m["content"] for m in recent] == ["m3", "m4"]


def test_turn_records_cost_and_token_counts(store: SessionStore):
    turn_id = store.add_turn("s1", "assistant", "hi", prompt_tokens=10,
                             completion_tokens=5, cost_usd=0.002)

    with store.db.session() as session:
        from tripmate.db import TurnRow
        row = session.get(TurnRow, turn_id)
        assert (row.prompt_tokens, row.cost_usd) == (10, 0.002)


def test_trace_events_are_persisted_against_their_turn(store: SessionStore):
    turn_id = store.add_turn("s1", "assistant", "hi")
    store.add_trace(turn_id, [
        TraceEvent(seq=1, event_type="LLM_CALL", timestamp=1.0, payload={"a": 1})
    ])

    with store.db.session() as session:
        from tripmate.db import TraceRow
        rows = session.query(TraceRow).filter_by(turn_id=turn_id).all()
        assert rows[0].event_type == "LLM_CALL"


def test_ensure_session_is_idempotent(store: SessionStore):
    store.ensure_session("s1")
    store.ensure_session("s1")

    with store.db.session() as session:
        from tripmate.db import SessionRow
        assert session.query(SessionRow).count() == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.db'`

- [ ] **Step 3: Implement `src/tripmate/db.py`**

```python
"""Persistence. SQLite by default, PostgreSQL via DATABASE_URL, no code change."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import (
    Float, ForeignKey, Integer, JSON, LargeBinary, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from tripmate.config import Settings, get_settings
from tripmate.models import TraceEvent

POOL_SIZE = 5
MAX_OVERFLOW = 10


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    last_active_at: Mapped[float] = mapped_column(Float, default=time.time)


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class TraceRow(Base):
    __tablename__ = "trace_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(Integer, ForeignKey("turns.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(32))
    timestamp: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class CacheRow(Base):
    __tablename__ = "query_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    answer: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        kwargs: dict[str, Any] = {"future": True}
        if not url.startswith("sqlite"):
            kwargs |= {
                "pool_size": POOL_SIZE,
                "max_overflow": MAX_OVERFLOW,
                "pool_pre_ping": True,
            }
        self.engine = create_engine(url, **kwargs)
        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class SessionStore:
    """Conversation and trace persistence."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure_session(self, session_id: str) -> None:
        with self.db.session() as session:
            if session.get(SessionRow, session_id) is None:
                session.add(SessionRow(id=session_id))

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
    ) -> int:
        self.ensure_session(session_id)
        with self.db.session() as session:
            row = TurnRow(
                session_id=session_id, role=role, content=content,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                cost_usd=cost_usd, latency_ms=latency_ms,
            )
            session.add(row)
            session.flush()
            return row.id

    def add_trace(self, turn_id: int, events: list[TraceEvent]) -> None:
        with self.db.session() as session:
            session.add_all([
                TraceRow(
                    turn_id=turn_id, seq=event.seq, event_type=event.event_type,
                    timestamp=event.timestamp, duration_ms=event.duration_ms,
                    payload=event.payload,
                )
                for event in events
            ])

    def history(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        """The most recent `limit` turns, oldest first, as LLM messages."""
        with self.db.session() as session:
            rows = (
                session.query(TurnRow)
                .filter_by(session_id=session_id)
                .order_by(TurnRow.id.desc())
                .limit(limit)
                .all()
            )
        return [{"role": row.role, "content": row.content} for row in reversed(rows)]


def build_database(settings: Settings | None = None) -> Database:
    settings = settings or get_settings()
    db = Database(url=settings.database_url)
    db.create_all()
    return db
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/db.py tests/test_db.py
git commit -m "feat: sqlalchemy persistence for sessions, turns and trace records"
```

---

### Task 11: Semantic query cache

**Files:**
- Create: `src/tripmate/core/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `Database`, `CacheRow` (Task 10), `Citation` (Task 2), settings (Task 1)
- Produces:
  - `CachedAnswer(answer: str, citations: list[Citation], similarity: float)`
  - `SemanticCache(db, embedder, threshold: float, enabled: bool = True)`
    - `lookup(query: str) -> CachedAnswer | None`
    - `store(query: str, answer: str, citations: list[Citation]) -> None`
  - `build_cache(db, settings=None) -> SemanticCache`
  - `cosine(a: list[float], b: list[float]) -> float`

The embedder is any object with `.embed(list[str]) -> Iterable[vector]`, so `fastembed`'s `TextEmbedding` and a stub both satisfy it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache.py
import pytest

from tripmate.core.cache import SemanticCache, cosine
from tripmate.db import Database
from tripmate.models import Citation


class StubEmbedder:
    """Maps known phrases to fixed vectors so similarity is exact and testable."""

    VECTORS = {
        "visa for japan": [1.0, 0.0, 0.0],
        "japan visa rules": [0.99, 0.141, 0.0],   # cosine ~0.99 with the above
        "what to pack for iceland": [0.0, 1.0, 0.0],
    }

    def embed(self, texts):
        return [self.VECTORS.get(text.strip().lower(), [0.0, 0.0, 1.0])
                for text in texts]


@pytest.fixture()
def cache(tmp_path) -> SemanticCache:
    db = Database(url=f"sqlite:///{tmp_path}/cache.db")
    db.create_all()
    return SemanticCache(db=db, embedder=StubEmbedder(), threshold=0.95)


def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_of_a_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_lookup_misses_on_an_empty_cache(cache: SemanticCache):
    assert cache.lookup("visa for japan") is None


def test_lookup_hits_on_the_exact_same_query(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.", [])
    assert cache.lookup("visa for japan").answer == "No visa needed."


def test_lookup_hits_on_a_semantically_similar_query(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.", [])
    hit = cache.lookup("japan visa rules")

    assert hit is not None
    assert hit.similarity >= 0.95


def test_lookup_misses_on_an_unrelated_query(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.", [])
    assert cache.lookup("what to pack for iceland") is None


def test_citations_survive_the_round_trip(cache: SemanticCache):
    cache.store("visa for japan", "No visa needed.",
                [Citation(city="tokyo", section="VISA & ENTRY")])

    assert cache.lookup("visa for japan").citations[0].ref == "tokyo/VISA & ENTRY"


def test_a_disabled_cache_never_hits(tmp_path):
    db = Database(url=f"sqlite:///{tmp_path}/off.db")
    db.create_all()
    disabled = SemanticCache(db=db, embedder=StubEmbedder(), threshold=0.95,
                             enabled=False)

    disabled.store("visa for japan", "answer", [])

    assert disabled.lookup("visa for japan") is None


def test_empty_query_is_never_cached(cache: SemanticCache):
    cache.store("   ", "answer", [])
    assert cache.lookup("   ") is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.core.cache'`

- [ ] **Step 3: Implement `src/tripmate/core/cache.py`**

```python
"""Semantic query cache.

Answers are replayed when a new query is close enough to one already answered.
This is the implemented answer to "how do you avoid redundant LLM calls", rather
than a paragraph about it.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Protocol

from tripmate.config import Settings, get_settings
from tripmate.db import CacheRow, Database
from tripmate.models import Citation

FLOAT_FORMAT = "f"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> Any: ...


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[Citation]
    similarity: float


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    magnitude = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / magnitude if magnitude else 0.0


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}{FLOAT_FORMAT}", *vector)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // struct.calcsize(FLOAT_FORMAT)
    return list(struct.unpack(f"<{count}{FLOAT_FORMAT}", blob))


class SemanticCache:
    def __init__(
        self, db: Database, embedder: Embedder, threshold: float, enabled: bool = True
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._threshold = threshold
        self._enabled = enabled

    def _embed(self, text: str) -> list[float]:
        vector = next(iter(self._embedder.embed([text])))
        return [float(value) for value in vector]

    def lookup(self, query: str) -> CachedAnswer | None:
        if not self._enabled or not query.strip():
            return None

        target = self._embed(query)
        with self._db.session() as session:
            rows = session.query(CacheRow).all()

        best: CachedAnswer | None = None
        for row in rows:
            similarity = cosine(target, _unpack(row.embedding))
            if similarity < self._threshold:
                continue
            if best is None or similarity > best.similarity:
                best = CachedAnswer(
                    answer=row.answer,
                    citations=[Citation(**c) for c in (row.citations or [])],
                    similarity=similarity,
                )
        return best

    def store(self, query: str, answer: str, citations: list[Citation]) -> None:
        if not self._enabled or not query.strip():
            return

        with self._db.session() as session:
            session.add(
                CacheRow(
                    query_text=query,
                    embedding=_pack(self._embed(query)),
                    answer=answer,
                    citations=[c.model_dump() for c in citations],
                )
            )


def build_cache(db: Database, settings: Settings | None = None) -> SemanticCache:
    from fastembed import TextEmbedding

    settings = settings or get_settings()
    return SemanticCache(
        db=db,
        embedder=TextEmbedding(model_name=settings.embedding_model),
        threshold=settings.semantic_cache_threshold,
        enabled=settings.semantic_cache_enabled,
    )
```

> Note: `lookup` scans every cached row. At this scale that is correct and simple.
> Past a few thousand entries it becomes a Chroma collection or a Redis vector index —
> recorded in the README's scalability section, not built now.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/tripmate/core/cache.py tests/test_cache.py
git commit -m "feat: semantic query cache replaying answers above similarity threshold"
```

---

### Task 12: Agent core

**Files:**
- Create: `src/tripmate/core/prompts.py`, `src/tripmate/core/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `ToolRegistry` (Task 3), `LLMClient`/`FakeLLMClient` (Task 8), `Tracer`/`EventType` (Task 9), `SessionStore` (Task 10), `SemanticCache` (Task 11), models (Task 2)
- Produces:
  - `SYSTEM_PROMPT: str`, `PROMPT_VERSION: str` in `prompts.py`
  - `validate_query(query: str, max_chars: int) -> str` — raises `InvalidQuery`
  - `InvalidQuery(Exception)`
  - `extract_citations(answer: str) -> list[Citation]`
  - `validate_citations(answer, citations, allowed_refs) -> tuple[str, list[Citation], list[str]]` — returns the cleaned answer, the kept citations, and the rejected refs
  - `Agent(llm, registry, tracer_factory, store=None, cache=None, settings=None)` with `chat(query: str, session_id: str | None = None) -> AgentResponse`

Loop invariants: never exceeds `max_tool_iterations`; independent tool calls in one turn run concurrently; no tool exception ever reaches the model.

- [ ] **Step 1: Implement `src/tripmate/core/prompts.py`**

```python
"""Versioned system prompt. Kept out of the loop so it can be diffed and evaluated."""

PROMPT_VERSION = "2026-09-16.1"

SYSTEM_PROMPT = """You are TripMate, a travel assistant.

You cover exactly four destinations: Tokyo, Reykjavik, Bangkok and Barcelona.

TOOLS
- search_destination_guide: visa and entry rules, best time to visit, local customs,
  packing tips, safety and health. Pass the `city` argument whenever the user names a
  destination, so the search is restricted to it.
- get_weather_forecast: expected conditions for a city on a date or during a month.

TOOL POLICY
- Visa, customs, safety, or "best time to visit" questions: use search_destination_guide.
- Weather or temperature questions: use get_weather_forecast.
- PACKING questions: use BOTH tools. Good packing advice needs the guide's tips and the
  actual conditions for that time of year. Call them together, then reconcile them.
- General conversation that needs no external facts: answer without tools.

GROUNDING
- Never state a travel fact that no tool returned. If a tool returns no data, say so
  plainly and name what you do cover.
- Cite every claim drawn from the destination guide as [city/SECTION], exactly as the
  tool result's `ref` field gives it, for example [tokyo/PACKING TIPS].
- When weather data comes back with source "climate_normal", describe it as typical
  conditions for that time of year, not as a forecast. When source is "mock_fallback",
  say the live weather service was unavailable and this is approximate offline data.

LIMITS
- You cannot book, reserve, cancel or pay for flights, hotels, tours or anything else.
  You cannot access accounts, itineraries or personal data. If asked, say clearly that
  you cannot do it and describe what you can help with instead. Never simulate or
  pretend to have performed such an action.
- For topics unrelated to travel, say they are outside what you cover.

AMBIGUITY
- If the destination or the timeframe is unclear and it changes the answer, ask one
  short clarifying question instead of guessing.

STYLE
- Be concise and concrete. Prefer short paragraphs or bullets over long prose.
"""
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_agent.py
import pytest

from tripmate.core.agent import (
    Agent, InvalidQuery, extract_citations, validate_citations, validate_query,
)
from tripmate.core.trace import EventType
from tripmate.models import Citation, LLMResponse, ToolCall, ToolResult
from tripmate.tools.registry import ToolRegistry, tool

from tests.fakes import FakeLLMClient
from typing import Annotated


@tool
def fake_guide(
    query: Annotated[str, "topic"],
    city: Annotated[str | None, "city"] = None,
) -> ToolResult:
    """Search the guide."""
    return ToolResult.ok("fake_guide", {"chunks": [
        {"ref": "tokyo/PACKING TIPS", "city": "tokyo", "section": "PACKING TIPS",
         "text": "Layered clothing works well.", "score": 0.8}
    ]})


@tool
def fake_weather(
    city: Annotated[str, "city"],
    date_or_month: Annotated[str, "month"],
) -> ToolResult:
    """Get weather."""
    return ToolResult.ok("fake_weather", {
        "city": "Tokyo", "period": "December", "source": "climate_normal",
        "conditions": "cold, mostly dry", "temp_range_c": [3.0, 12.0],
        "precip_days": 4,
    })


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(fake_guide)
    registry.register(fake_weather)
    return registry


def _agent(script: list[LLMResponse]) -> tuple[Agent, FakeLLMClient]:
    fake = FakeLLMClient(script)
    return Agent(llm=fake, registry=_registry()), fake


# --- input validation ---

def test_validate_query_strips_surrounding_whitespace():
    assert validate_query("  hello  ", max_chars=100) == "hello"


def test_validate_query_rejects_empty_input():
    with pytest.raises(InvalidQuery, match="empty"):
        validate_query("   ", max_chars=100)


def test_validate_query_rejects_overlong_input():
    with pytest.raises(InvalidQuery, match="too long"):
        validate_query("x" * 101, max_chars=100)


# --- citations ---

def test_extract_citations_finds_every_reference():
    refs = extract_citations("See [tokyo/PACKING TIPS] and [tokyo/SAFETY & HEALTH].")
    assert [c.ref for c in refs] == ["tokyo/PACKING TIPS", "tokyo/SAFETY & HEALTH"]


def test_validate_citations_keeps_references_that_were_retrieved():
    answer = "Pack layers [tokyo/PACKING TIPS]."
    cleaned, kept, rejected = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert kept[0].ref == "tokyo/PACKING TIPS"
    assert rejected == []
    assert cleaned == answer


def test_validate_citations_strips_references_never_retrieved():
    answer = "Pack layers [osaka/PACKING TIPS]."
    cleaned, kept, rejected = validate_citations(
        answer, extract_citations(answer), {"tokyo/PACKING TIPS"}
    )
    assert kept == []
    assert rejected == ["osaka/PACKING TIPS"]
    assert "[osaka/PACKING TIPS]" not in cleaned


# --- the loop ---

def test_no_tool_query_returns_the_model_answer_directly():
    agent, _ = _agent([LLMResponse(content="I help with travel questions.")])
    response = agent.chat("what can you do?")

    assert response.answer == "I help with travel questions."


def test_no_tool_query_makes_exactly_one_llm_call():
    agent, fake = _agent([LLMResponse(content="hello")])
    agent.chat("hi")

    assert len(fake.calls) == 1


def test_single_tool_query_calls_that_tool_then_answers():
    agent, _ = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_guide",
                                         arguments={"query": "visa", "city": "tokyo"})]),
        LLMResponse(content="No visa needed [tokyo/PACKING TIPS]."),
    ])
    response = agent.chat("do I need a visa for Tokyo?")

    tools_called = [e.payload["tool"] for e in response.trace
                    if e.event_type == EventType.TOOL_CALL]
    assert tools_called == ["fake_guide"]


def test_multi_tool_query_calls_both_tools_in_one_turn():
    agent, _ = _agent([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="fake_guide",
                     arguments={"query": "packing", "city": "tokyo"}),
            ToolCall(id="2", name="fake_weather",
                     arguments={"city": "Tokyo", "date_or_month": "December"}),
        ]),
        LLMResponse(content="Pack warm layers [tokyo/PACKING TIPS]."),
    ])
    response = agent.chat("what should I pack for Tokyo in December?")

    tools_called = {e.payload["tool"] for e in response.trace
                    if e.event_type == EventType.TOOL_CALL}
    assert tools_called == {"fake_guide", "fake_weather"}


def test_tool_results_are_appended_as_tool_role_messages():
    agent, fake = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_weather",
                                         arguments={"city": "Tokyo",
                                                    "date_or_month": "December"})]),
        LLMResponse(content="Cold."),
    ])
    agent.chat("weather in Tokyo in December?")

    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert "tool" in roles


def test_valid_citations_are_returned_on_the_response():
    agent, _ = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_guide",
                                         arguments={"query": "packing"})]),
        LLMResponse(content="Layers [tokyo/PACKING TIPS]."),
    ])
    response = agent.chat("packing for Tokyo?")

    assert [c.ref for c in response.citations] == ["tokyo/PACKING TIPS"]


def test_hallucinated_citations_are_stripped_and_traced():
    agent, _ = _agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="fake_guide",
                                         arguments={"query": "packing"})]),
        LLMResponse(content="Layers [paris/PACKING TIPS]."),
    ])
    response = agent.chat("packing for Paris?")

    assert response.citations == []
    assert "[paris/PACKING TIPS]" not in response.answer
    assert any(e.event_type == EventType.CITATION_REJECTED for e in response.trace)


def test_iteration_ceiling_forces_a_final_answer():
    looping = [
        LLMResponse(tool_calls=[ToolCall(id=str(i), name="fake_weather",
                                         arguments={"city": "Tokyo",
                                                    "date_or_month": "December"})])
        for i in range(10)
    ]
    fake = FakeLLMClient(looping)
    agent = Agent(llm=fake, registry=_registry())

    response = agent.chat("weather?")

    assert response.answer  # non-empty, synthesised from what was gathered
    assert len(fake.calls) <= 6  # max_tool_iterations (5) plus the forced synthesis


def test_empty_input_is_rejected_before_any_llm_call():
    fake = FakeLLMClient([])
    agent = Agent(llm=fake, registry=_registry())

    response = agent.chat("   ")

    assert "empty" in response.answer.lower()
    assert fake.calls == []


def test_trace_records_tokens_and_cost_from_llm_calls():
    agent, _ = _agent([
        LLMResponse(content="hi", prompt_tokens=100, completion_tokens=20,
                    cost_usd=0.001)
    ])
    response = agent.chat("hello")

    assert response.prompt_tokens == 100
    assert response.cost_usd == pytest.approx(0.001)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.core.agent'`

- [ ] **Step 4: Implement `src/tripmate/core/agent.py`**

```python
"""The orchestration loop.

One loop, provider-agnostic: the model chooses tools, the registry executes them,
results go back as tool messages, and the model synthesises a grounded answer.
"""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from tripmate.config import Settings, get_settings
from tripmate.core.cache import SemanticCache
from tripmate.core.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from tripmate.core.trace import EventType, Tracer
from tripmate.db import SessionStore
from tripmate.models import (
    AgentResponse, Citation, LLMResponse, ToolCall, ToolResult,
)
from tripmate.models import CITATION_RE
from tripmate.tools.registry import ToolRegistry

FORCED_SYNTHESIS_NOTE = (
    "You have gathered enough tool output. Answer the user's question now, "
    "using only what the tools returned. Do not request any more tools."
)


class InvalidQuery(Exception):
    """Raised when input fails validation at the boundary."""


class SupportsComplete(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse: ...


def validate_query(query: str, max_chars: int) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        raise InvalidQuery("Your message looks empty. Ask me about Tokyo, Reykjavik, "
                           "Bangkok or Barcelona.")
    if len(cleaned) > max_chars:
        raise InvalidQuery(f"That message is too long ({len(cleaned)} characters). "
                           f"Please keep it under {max_chars}.")
    return cleaned


def extract_citations(answer: str) -> list[Citation]:
    return [
        Citation(city=match.group(1).strip().lower(),
                 section=match.group(2).strip().upper())
        for match in CITATION_RE.finditer(answer or "")
    ]


def validate_citations(
    answer: str, citations: list[Citation], allowed_refs: set[str]
) -> tuple[str, list[Citation], list[str]]:
    """Strip any citation that does not correspond to a chunk actually retrieved."""
    kept: list[Citation] = []
    rejected: list[str] = []
    cleaned = answer

    for citation in citations:
        if citation.ref in allowed_refs:
            if citation.ref not in {c.ref for c in kept}:
                kept.append(citation)
            continue
        rejected.append(citation.ref)
        cleaned = cleaned.replace(f"[{citation.ref}]", "")
        cleaned = cleaned.replace(f"[{citation.city}/{citation.section.title()}]", "")

    return cleaned.replace("  ", " ").strip(), kept, rejected


def _refs_from(result: ToolResult) -> set[str]:
    if result.status != "ok" or not result.data:
        return set()
    return {chunk["ref"] for chunk in result.data.get("chunks", []) if "ref" in chunk}


class Agent:
    def __init__(
        self,
        llm: SupportsComplete,
        registry: ToolRegistry,
        tracer_factory: Callable[[str], Tracer] | None = None,
        store: SessionStore | None = None,
        cache: SemanticCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._settings = settings or get_settings()
        self._store = store
        self._cache = cache
        self._tracer_factory = tracer_factory or (lambda sid: Tracer(session_id=sid))

    def chat(self, query: str, session_id: str | None = None) -> AgentResponse:
        session_id = session_id or uuid.uuid4().hex[:12]
        tracer = self._tracer_factory(session_id)
        started = time.perf_counter()

        try:
            cleaned = validate_query(query, self._settings.max_query_chars)
        except InvalidQuery as exc:
            tracer.record(EventType.QUERY_RECEIVED, valid=False, reason=str(exc))
            return self._finish(str(exc), [], tracer, session_id, started)

        tracer.record(EventType.QUERY_RECEIVED, query=cleaned,
                      prompt_version=PROMPT_VERSION)

        cached = self._cache.lookup(cleaned) if self._cache else None
        if cached is not None:
            tracer.record(EventType.CACHE_HIT, similarity=round(cached.similarity, 4))
            return self._finish(cached.answer, cached.citations, tracer, session_id,
                                started, was_cached=True)

        answer, citations = self._run_loop(cleaned, session_id, tracer)

        if self._cache:
            self._cache.store(cleaned, answer, citations)

        return self._finish(answer, citations, tracer, session_id, started)

    def _run_loop(
        self, query: str, session_id: str, tracer: Tracer
    ) -> tuple[str, list[Citation]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self._store:
            messages.extend(self._store.history(session_id))
        messages.append({"role": "user", "content": query})

        allowed_refs: set[str] = set()
        schemas = self._registry.schemas()

        for iteration in range(self._settings.max_tool_iterations):
            is_last = iteration == self._settings.max_tool_iterations - 1
            if is_last:
                messages.append({"role": "system", "content": FORCED_SYNTHESIS_NOTE})

            response = self._call_llm(messages, None if is_last else schemas, tracer)

            if not response.tool_calls:
                return self._finalise(response.content or "", allowed_refs, tracer)

            messages.append(self._assistant_message(response))
            results = self._dispatch_all(response.tool_calls, tracer)

            for call, result in zip(response.tool_calls, results):
                allowed_refs |= _refs_from(result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(result.model_dump(), default=str),
                })

        final = self._call_llm(messages, None, tracer)
        return self._finalise(final.content or "", allowed_refs, tracer)

    def _call_llm(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        started = time.perf_counter()
        response = self._llm.complete(messages, tools=tools)
        tracer.record(
            EventType.LLM_CALL,
            duration_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
            requested_tools=[c.name for c in response.tool_calls],
        )
        return response

    @staticmethod
    def _assistant_message(response: LLMResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name,
                                 "arguments": json.dumps(call.arguments)},
                }
                for call in response.tool_calls
            ],
        }

    def _dispatch_all(
        self, calls: list[ToolCall], tracer: Tracer
    ) -> list[ToolResult]:
        """Run independent tool calls concurrently; a tool never raises into the loop."""
        for call in calls:
            tracer.record(EventType.TOOL_CALL, tool=call.name, arguments=call.arguments)

        def run(call: ToolCall) -> tuple[ToolResult, float]:
            started = time.perf_counter()
            result = self._registry.dispatch(call.name, call.arguments)
            return result, (time.perf_counter() - started) * 1000

        if len(calls) == 1:
            outcomes = [run(calls[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                outcomes = list(pool.map(run, calls))

        results: list[ToolResult] = []
        for call, (result, duration) in zip(calls, outcomes):
            event_type = (
                EventType.TOOL_ERROR if result.status == "error" else EventType.TOOL_RESULT
            )
            tracer.record(event_type, duration_ms=duration, tool=call.name,
                          status=result.status, reason=result.reason)
            if (result.data or {}).get("source") == "mock_fallback":
                tracer.record(EventType.FALLBACK_USED, tool=call.name)
            results.append(result)
        return results

    @staticmethod
    def _finalise(
        raw_answer: str, allowed_refs: set[str], tracer: Tracer
    ) -> tuple[str, list[Citation]]:
        answer, kept, rejected = validate_citations(
            raw_answer, extract_citations(raw_answer), allowed_refs
        )
        if rejected:
            tracer.record(EventType.CITATION_REJECTED, refs=rejected)
        tracer.record(EventType.ANSWER_SYNTHESIZED,
                      citations=[c.ref for c in kept], length=len(answer))
        return answer, kept

    def _finish(
        self, answer: str, citations: list[Citation], tracer: Tracer,
        session_id: str, started: float, was_cached: bool = False,
    ) -> AgentResponse:
        totals = tracer.totals()
        latency_ms = (time.perf_counter() - started) * 1000
        tracer.flush()
        return AgentResponse(
            answer=answer, citations=citations, trace=tracer.events,
            prompt_tokens=totals.prompt_tokens,
            completion_tokens=totals.completion_tokens,
            cost_usd=totals.cost_usd, latency_ms=latency_ms,
            session_id=session_id, was_cached=was_cached,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: 17 passed

- [ ] **Step 6: Run the whole suite to check nothing regressed**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/tripmate/core/prompts.py src/tripmate/core/agent.py tests/test_agent.py
git commit -m "feat: agent loop with parallel tool dispatch and citation validation"
```

---

### Task 13: Tool-selection tests

**Files:**
- Create: `tests/test_tool_selection.py`
- Test: itself

**Interfaces:**
- Consumes: `Agent` (Task 12), real tools (Tasks 6-7), `FakeLLMClient` (Task 8)
- Produces: a reusable `build_test_agent(script)` helper importable by Tasks 14-15

These tests pin the brief's Module 1 requirement — dynamic per-query selection of single-tool, multi-tool, and no-tool paths — using the **real** tools with a scripted model, so tool wiring is exercised without provider flake.

- [ ] **Step 1: Write the test file**

```python
# tests/test_tool_selection.py
from pathlib import Path

import pytest

from tripmate.core.agent import Agent
from tripmate.core.trace import EventType
from tripmate.models import LLMResponse, ToolCall
from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import (
    reset_store, search_destination_guide, set_store,
)
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import clear_cache, get_weather_forecast

from tests.fakes import FakeLLMClient


@pytest.fixture(scope="module", autouse=True)
def _real_store(tmp_path_factory):
    store = ChromaStore(
        path=str(tmp_path_factory.mktemp("chroma_sel")),
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    store.add(load_all(Path("data/destinations")))
    set_store(store)
    clear_cache()
    yield
    reset_store()
    clear_cache()


def build_test_agent(script: list[LLMResponse]) -> tuple[Agent, FakeLLMClient]:
    """Real tools, scripted model. Shared by the error and integration suites."""
    registry = ToolRegistry()
    registry.register(search_destination_guide)
    registry.register(get_weather_forecast)
    fake = FakeLLMClient(script)
    return Agent(llm=fake, registry=registry), fake


def tools_called(response) -> list[str]:
    return [e.payload["tool"] for e in response.trace
            if e.event_type == EventType.TOOL_CALL]


# --- no tool ---

def test_greeting_uses_no_tools():
    agent, _ = build_test_agent([LLMResponse(content="Hello! Ask me about Tokyo.")])
    assert tools_called(agent.chat("hello there")) == []


def test_capability_question_uses_no_tools():
    agent, _ = build_test_agent([
        LLMResponse(content="I cover Tokyo, Reykjavik, Bangkok and Barcelona.")
    ])
    assert tools_called(agent.chat("what can you help with?")) == []


# --- single tool: RAG ---

def test_visa_question_uses_only_the_guide():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "visa and entry requirements", "city": "tokyo"})]),
        LLMResponse(content="Many nationalities enter visa-free "
                            "[tokyo/VISA & ENTRY]."),
    ])
    assert tools_called(agent.chat("do I need a visa for Japan?")) == [
        "search_destination_guide"
    ]


def test_guide_tool_result_reaches_the_second_llm_call():
    agent, fake = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "local customs", "city": "tokyo"})]),
        LLMResponse(content="Bowing is common [tokyo/LOCAL CUSTOMS]."),
    ])
    agent.chat("what are the customs in Tokyo?")

    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert "Bowing" in tool_messages[0]["content"]


# --- single tool: weather ---

def test_weather_question_uses_only_the_weather_tool():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="get_weather_forecast",
            arguments={"city": "Reykjavik", "date_or_month": "January"})]),
        LLMResponse(content="Cold, around -3 to 3 C."),
    ])
    assert tools_called(agent.chat("how cold is Reykjavik in January?")) == [
        "get_weather_forecast"
    ]


# --- multi tool ---

def test_packing_question_uses_both_tools():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="search_destination_guide",
                     arguments={"query": "packing tips", "city": "tokyo"}),
            ToolCall(id="2", name="get_weather_forecast",
                     arguments={"city": "Tokyo", "date_or_month": "December"}),
        ]),
        LLMResponse(content="Pack warm layers [tokyo/PACKING TIPS]."),
    ])
    assert set(tools_called(agent.chat("what should I pack for Tokyo in December?"))) == {
        "search_destination_guide", "get_weather_forecast"
    }


def test_both_tool_results_are_present_before_synthesis():
    agent, fake = build_test_agent([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="search_destination_guide",
                     arguments={"query": "packing tips", "city": "bangkok"}),
            ToolCall(id="2", name="get_weather_forecast",
                     arguments={"city": "Bangkok", "date_or_month": "July"}),
        ]),
        LLMResponse(content="Light, breathable clothing [bangkok/PACKING TIPS]."),
    ])
    agent.chat("what should I pack for Bangkok in July?")

    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2


def test_sequential_tool_calls_across_two_iterations_are_supported():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "best time to visit", "city": "barcelona"})]),
        LLMResponse(tool_calls=[ToolCall(
            id="2", name="get_weather_forecast",
            arguments={"city": "Barcelona", "date_or_month": "May"})]),
        LLMResponse(content="May is mild [barcelona/BEST TIME TO VISIT]."),
    ])
    response = agent.chat("when should I visit Barcelona and how warm is it then?")

    assert tools_called(response) == [
        "search_destination_guide", "get_weather_forecast"
    ]
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_tool_selection.py -v`
Expected: 8 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_selection.py
git commit -m "test: tool selection coverage for single, multi and no-tool queries"
```

---

### Task 14: Error scenario tests

**Files:**
- Create: `tests/test_error_scenarios.py`

**Interfaces:**
- Consumes: `build_test_agent` (Task 13)
- Produces: one test per bullet in the brief's Error Handling list, named after it

- [ ] **Step 1: Write the test file**

```python
# tests/test_error_scenarios.py
"""One test per error scenario named in the assessment brief.

Brief: unknown destination, missing weather data, ambiguous query, tool failure or
timeout, out-of-scope request, malformed or empty input.
"""

import httpx
import pytest
import respx

from tripmate.core.trace import EventType
from tripmate.models import LLMResponse, ToolCall
from tripmate.tools.weather import GEOCODE_URL, clear_cache

from tests.fakes import FakeLLMClient
from tests.test_tool_selection import build_test_agent  # noqa: F401 (fixture import)
from tests.test_tool_selection import _real_store  # noqa: F401


@pytest.fixture(autouse=True)
def _clear_weather_cache():
    clear_cache()
    yield
    clear_cache()


# --- 1. Unknown / unsupported destination ---

def test_unknown_destination_returns_no_data_and_names_supported_cities():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "visa requirements", "city": "paris"})]),
        LLMResponse(content="I don't have a guide for Paris. I cover Tokyo, "
                            "Reykjavik, Bangkok and Barcelona."),
    ])
    response = agent.chat("what are the visa rules for Paris?")

    statuses = [e.payload["status"] for e in response.trace
                if e.event_type == EventType.TOOL_RESULT]
    assert statuses == ["no_data"]
    assert "Paris" in response.answer


def test_unknown_destination_answer_carries_no_citations():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="search_destination_guide",
            arguments={"query": "visa", "city": "paris"})]),
        LLMResponse(content="No guide for Paris."),
    ])
    assert agent.chat("visa rules for Paris?").citations == []


# --- 2. Missing / incomplete weather data ---

@respx.mock
def test_unresolvable_city_returns_no_data_from_the_weather_tool():
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="get_weather_forecast",
            arguments={"city": "Atlantis", "date_or_month": "July"})]),
        LLMResponse(content="I couldn't find weather data for Atlantis."),
    ])
    response = agent.chat("weather in Atlantis in July?")

    statuses = [e.payload["status"] for e in response.trace
                if e.event_type == EventType.TOOL_RESULT]
    assert statuses == ["no_data"]


def test_unparseable_timeframe_returns_no_data():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="get_weather_forecast",
            arguments={"city": "Tokyo", "date_or_month": "whenever"})]),
        LLMResponse(content="Could you tell me which month you mean?"),
    ])
    response = agent.chat("weather in Tokyo whenever?")

    statuses = [e.payload["status"] for e in response.trace
                if e.event_type == EventType.TOOL_RESULT]
    assert statuses == ["no_data"]


# --- 3. Ambiguous query ---

def test_ambiguous_query_asks_for_clarification_without_calling_tools():
    agent, _ = build_test_agent([
        LLMResponse(content="Which destination did you have in mind — Tokyo, "
                            "Reykjavik, Bangkok or Barcelona?")
    ])
    response = agent.chat("what should I pack?")

    assert [e for e in response.trace if e.event_type == EventType.TOOL_CALL] == []
    assert "?" in response.answer


# --- 4. Tool failure or timeout ---

@respx.mock
def test_weather_api_timeout_falls_back_and_records_the_fallback():
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="get_weather_forecast",
            arguments={"city": "Tokyo", "date_or_month": "December"})]),
        LLMResponse(content="The live weather service was unavailable; typical "
                            "December conditions in Tokyo are cold and dry."),
    ])
    response = agent.chat("weather in Tokyo in December?")

    assert any(e.event_type == EventType.FALLBACK_USED for e in response.trace)


@respx.mock
def test_weather_failure_without_offline_data_surfaces_a_tool_error():
    respx.get(GEOCODE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(
            id="1", name="get_weather_forecast",
            arguments={"city": "Lisbon", "date_or_month": "December"})]),
        LLMResponse(content="I couldn't reach the weather service for Lisbon."),
    ])
    response = agent.chat("weather in Lisbon in December?")

    assert any(e.event_type == EventType.TOOL_ERROR for e in response.trace)


def test_unknown_tool_name_from_the_model_becomes_a_tool_error():
    agent, _ = build_test_agent([
        LLMResponse(tool_calls=[ToolCall(id="1", name="book_flight",
                                         arguments={"to": "Tokyo"})]),
        LLMResponse(content="I can't book flights."),
    ])
    response = agent.chat("book me a flight")

    assert any(e.event_type == EventType.TOOL_ERROR for e in response.trace)


# --- 5. Out-of-scope request ---

def test_booking_request_is_refused_without_calling_tools():
    agent, _ = build_test_agent([
        LLMResponse(content="I can't book flights. I can tell you about visas, "
                            "weather, packing, customs and safety for four cities.")
    ])
    response = agent.chat("can you book my flight to Barcelona?")

    assert [e for e in response.trace if e.event_type == EventType.TOOL_CALL] == []
    assert "can't" in response.answer.lower()


def test_unrelated_topic_is_declined_without_calling_tools():
    agent, _ = build_test_agent([
        LLMResponse(content="That's outside what I cover. I help with travel "
                            "questions for four destinations.")
    ])
    response = agent.chat("write me a python script to sort a list")

    assert [e for e in response.trace if e.event_type == EventType.TOOL_CALL] == []


# --- 6. Malformed or empty input ---

def test_empty_input_is_rejected_before_any_llm_call():
    registry_agent, fake = build_test_agent([])
    response = registry_agent.chat("")

    assert fake.calls == []
    assert "empty" in response.answer.lower()


def test_whitespace_only_input_is_rejected_before_any_llm_call():
    agent, fake = build_test_agent([])
    agent.chat("\n\t   ")

    assert fake.calls == []


def test_overlong_input_is_rejected_before_any_llm_call():
    agent, fake = build_test_agent([])
    response = agent.chat("x" * 5000)

    assert fake.calls == []
    assert "too long" in response.answer.lower()
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_error_scenarios.py -v`
Expected: 13 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_error_scenarios.py
git commit -m "test: one named test per error scenario in the brief"
```

---

### Task 15: End-to-end multi-tool integration test

**Files:**
- Create: `tests/test_integration_multitool.py`

**Interfaces:**
- Consumes: `Agent` (Task 12), real tools, real `ChromaStore`, real `Database`, real `SemanticCache`, `FakeLLMClient`

Everything is real except the model. This is the closest thing to a live run that stays deterministic in CI.

- [ ] **Step 1: Write the test file**

```python
# tests/test_integration_multitool.py
"""Full multi-tool flow: real RAG, real weather (mocked HTTP), real DB, real cache."""

from pathlib import Path

import httpx
import pytest
import respx

from tripmate.core.agent import Agent
from tripmate.core.cache import SemanticCache
from tripmate.core.trace import EventType, Tracer
from tripmate.db import Database, SessionStore
from tripmate.models import LLMResponse, ToolCall
from tripmate.rag.chunker import load_all
from tripmate.rag.store import ChromaStore
from tripmate.tools.destination import reset_store, search_destination_guide, set_store
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import ARCHIVE_URL, GEOCODE_URL, clear_cache
from tripmate.tools.weather import get_weather_forecast

from tests.fakes import FakeLLMClient

GEOCODE_TOKYO = {"results": [{"latitude": 35.68, "longitude": 139.75, "name": "Tokyo"}]}
ARCHIVE_DECEMBER = {
    "daily": {
        "time": ["2021-12-01", "2021-12-02", "2022-12-01", "2022-12-02"],
        "temperature_2m_max": [12.0, 11.0, 13.0, 10.0],
        "temperature_2m_min": [4.0, 3.0, 5.0, 2.0],
        "precipitation_sum": [0.0, 0.0, 2.0, 0.0],
    }
}

PACKING_SCRIPT = [
    LLMResponse(
        tool_calls=[
            ToolCall(id="1", name="search_destination_guide",
                     arguments={"query": "packing tips", "city": "tokyo"}),
            ToolCall(id="2", name="get_weather_forecast",
                     arguments={"city": "Tokyo", "date_or_month": "December"}),
        ],
        prompt_tokens=400, completion_tokens=60, cost_usd=0.0008,
    ),
    LLMResponse(
        content="Pack warm layers and comfortable walking shoes "
                "[tokyo/PACKING TIPS]. December in Tokyo is typically cold and "
                "mostly dry, roughly 3 to 12 C.",
        prompt_tokens=900, completion_tokens=80, cost_usd=0.0015,
    ),
]


@pytest.fixture()
def agent(tmp_path):
    store = ChromaStore(path=str(tmp_path / "chroma"),
                        embedding_model="BAAI/bge-small-en-v1.5")
    store.add(load_all(Path("data/destinations")))
    set_store(store)
    clear_cache()

    db = Database(url=f"sqlite:///{tmp_path}/it.db")
    db.create_all()

    registry = ToolRegistry()
    registry.register(search_destination_guide)
    registry.register(get_weather_forecast)

    built = Agent(
        llm=FakeLLMClient(PACKING_SCRIPT),
        registry=registry,
        tracer_factory=lambda sid: Tracer(session_id=sid,
                                          trace_dir=str(tmp_path / "traces")),
        store=SessionStore(db),
        cache=SemanticCache(db=db, embedder=store._embedder, threshold=0.95),
    )
    yield built
    reset_store()
    clear_cache()


@respx.mock
def test_packing_query_calls_both_tools_and_synthesises_one_answer(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    tools = {e.payload["tool"] for e in response.trace
             if e.event_type == EventType.TOOL_CALL}
    assert tools == {"search_destination_guide", "get_weather_forecast"}
    assert "layers" in response.answer


@respx.mock
def test_answer_cites_a_chunk_that_was_actually_retrieved(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    assert [c.ref for c in response.citations] == ["tokyo/PACKING TIPS"]


@respx.mock
def test_cost_and_tokens_accumulate_across_both_llm_calls(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    assert response.prompt_tokens == 1300
    assert response.cost_usd == pytest.approx(0.0023)


@respx.mock
def test_trace_is_written_to_disk_as_jsonl(agent, tmp_path):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    response = agent.chat("What should I pack for Tokyo in December?")

    written = list((tmp_path / "traces").glob("*.jsonl"))
    assert written and response.session_id in written[0].name


@respx.mock
def test_repeating_the_query_is_served_from_the_semantic_cache(agent):
    respx.get(GEOCODE_URL).mock(return_value=httpx.Response(200, json=GEOCODE_TOKYO))
    respx.get(ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=ARCHIVE_DECEMBER))

    first = agent.chat("What should I pack for Tokyo in December?")
    second = agent.chat("What should I pack for Tokyo in December?")

    assert second.was_cached is True
    assert second.answer == first.answer
    assert any(e.event_type == EventType.CACHE_HIT for e in second.trace)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_integration_multitool.py -v`
Expected: 5 passed

The cache test is the reason the script has only two entries: a second uncached run would exhaust `FakeLLMClient` and raise `LLMError`. That exhaustion is the proof the cache prevented the LLM calls.

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_multitool.py
git commit -m "test: end-to-end multi-tool integration with real rag, db and cache"
```

---

### Task 16: CLI adapter

**Files:**
- Create: `src/tripmate/adapters/cli.py`, `src/tripmate/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: everything built so far
- Produces:
  - `build_agent(settings=None) -> Agent` in `bootstrap.py` — the single wiring point shared by both adapters
  - `render_trace(console, events) -> None` in `cli.py`
  - `main() -> None` — the `tripmate` console script

`bootstrap.build_agent` exists so CLI and API cannot drift apart: both get the identical object.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bootstrap.py
from tripmate.bootstrap import build_agent
from tripmate.config import Settings


def test_build_agent_registers_both_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        database_url=f"sqlite:///{tmp_path}/boot.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    agent = build_agent(settings=settings)

    assert set(agent._registry.names()) == {
        "search_destination_guide", "get_weather_forecast"
    }


def test_build_agent_ingests_the_data_pack(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = Settings(
        llm_api_key="test-key",
        chroma_path=str(tmp_path / "chroma2"),
        database_url=f"sqlite:///{tmp_path}/boot2.db",
        trace_dir=str(tmp_path / "traces"),
        semantic_cache_enabled=False,
    )

    build_agent(settings=settings)

    from tripmate.rag.store import build_store
    assert build_store(settings).count() == 20
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.bootstrap'`

- [ ] **Step 3: Implement `src/tripmate/bootstrap.py`**

```python
"""Single wiring point. Both adapters build the agent from here, so they cannot drift."""

from __future__ import annotations

from tripmate.config import Settings, get_settings
from tripmate.core.agent import Agent
from tripmate.core.cache import build_cache
from tripmate.core.trace import Tracer
from tripmate.db import SessionStore, build_database
from tripmate.llm.client import build_llm_client
from tripmate.rag.ingest import ingest
from tripmate.rag.store import build_store
from tripmate.tools.destination import search_destination_guide, set_store
from tripmate.tools.registry import ToolRegistry
from tripmate.tools.weather import get_weather_forecast


def build_agent(settings: Settings | None = None) -> Agent:
    settings = settings or get_settings()

    ingest(settings=settings)          # idempotent: a no-op once the pack is loaded
    set_store(build_store(settings))

    registry = ToolRegistry()
    registry.register(search_destination_guide)
    registry.register(get_weather_forecast)

    database = build_database(settings)

    return Agent(
        llm=build_llm_client(settings),
        registry=registry,
        tracer_factory=lambda sid: Tracer(session_id=sid, trace_dir=settings.trace_dir),
        store=SessionStore(database),
        cache=build_cache(database, settings),
        settings=settings,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: 2 passed

- [ ] **Step 5: Implement `src/tripmate/adapters/cli.py`**

```python
"""Multi-turn CLI with live reasoning-trace rendering."""

from __future__ import annotations

import uuid

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tripmate.bootstrap import build_agent
from tripmate.config import ConfigError, get_settings
from tripmate.core.trace import EventType
from tripmate.models import AgentResponse, TraceEvent

BANNER = (
    "TripMate — ask about Tokyo, Reykjavik, Bangkok or Barcelona.\n"
    "Commands: /trace  /cost  /reset  /quit"
)

EVENT_STYLES = {
    EventType.QUERY_RECEIVED: "dim",
    EventType.CACHE_HIT: "bold green",
    EventType.LLM_CALL: "cyan",
    EventType.TOOL_CALL: "yellow",
    EventType.TOOL_RESULT: "green",
    EventType.TOOL_ERROR: "bold red",
    EventType.FALLBACK_USED: "bold magenta",
    EventType.CITATION_REJECTED: "bold red",
    EventType.ANSWER_SYNTHESIZED: "dim",
}


def render_trace(console: Console, events: list[TraceEvent]) -> None:
    table = Table(title="reasoning trace", show_lines=False, title_style="dim")
    table.add_column("#", justify="right", width=3)
    table.add_column("event", width=20)
    table.add_column("ms", justify="right", width=7)
    table.add_column("detail", overflow="fold")

    for event in events:
        duration = f"{event.duration_ms:.0f}" if event.duration_ms else ""
        detail = ", ".join(
            f"{key}={value}" for key, value in event.payload.items() if value not in (None, [], {})
        )
        table.add_row(
            str(event.seq),
            f"[{EVENT_STYLES.get(event.event_type, 'white')}]{event.event_type}[/]",
            duration,
            detail[:160],
        )
    console.print(table)


def _render_answer(console: Console, response: AgentResponse) -> None:
    console.print(Panel(response.answer, title="TripMate", border_style="blue"))
    refs = ", ".join(c.ref for c in response.citations) or "none"
    cached = " (cached)" if response.was_cached else ""
    console.print(
        f"[dim]sources: {refs} | {response.prompt_tokens}+"
        f"{response.completion_tokens} tokens | ${response.cost_usd:.5f} | "
        f"{response.latency_ms:.0f} ms{cached}[/dim]\n"
    )


def main() -> None:
    console = Console()
    try:
        settings = get_settings()
        agent = build_agent(settings)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        return

    console.print(Panel(BANNER, border_style="blue"))
    console.print(f"[dim]model: {settings.llm_model}[/dim]\n")

    session_id = uuid.uuid4().hex[:12]
    last: AgentResponse | None = None
    total_cost = 0.0

    while True:
        try:
            query = console.input("[bold]you >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            return

        if query in {"/quit", "/exit"}:
            console.print("bye")
            return
        if query == "/reset":
            session_id = uuid.uuid4().hex[:12]
            console.print("[dim]new session[/dim]\n")
            continue
        if query == "/cost":
            console.print(f"[dim]session total: ${total_cost:.5f}[/dim]\n")
            continue
        if query == "/trace":
            if last is None:
                console.print("[dim]no turn yet[/dim]\n")
            else:
                render_trace(console, last.trace)
            continue

        with console.status("thinking..."):
            last = agent.chat(query, session_id=session_id)

        total_cost += last.cost_usd
        render_trace(console, last.trace)
        _render_answer(console, last)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Manually verify the CLI against a real provider**

```bash
cp .env.example .env
# Edit .env: set LLM_MODEL and LLM_API_KEY (or LLM_MODEL=ollama/qwen3.5 with no key)
uv run tripmate
```

Try each of these and confirm the trace shows the expected tools:

| Query | Expect |
|---|---|
| `Do I need a visa for Japan?` | one `TOOL_CALL search_destination_guide` |
| `How cold is Reykjavik in January?` | one `TOOL_CALL get_weather_forecast` |
| `What should I pack for Tokyo in December?` | **both** tools |
| `Can you book my flight?` | no tool calls, explicit refusal |
| `What about Bangkok?` | follow-up resolves from history |

- [ ] **Step 7: Commit**

```bash
git add src/tripmate/bootstrap.py src/tripmate/adapters/cli.py tests/test_bootstrap.py
git commit -m "feat: cli adapter with live trace rendering and shared bootstrap"
```

---

### Task 17: FastAPI adapter

**Files:**
- Create: `src/tripmate/adapters/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `build_agent` (Task 16), `SessionStore` (Task 10)
- Produces:
  - `create_app(agent=None, store=None) -> FastAPI`
  - `app` — module-level instance for `uvicorn tripmate.adapters.api:app`
  - Request/response models `ChatRequest`, `ChatResponse`, `HealthResponse`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

from tripmate.adapters.api import create_app
from tripmate.core.agent import Agent
from tripmate.models import LLMResponse, ToolCall
from tripmate.tools.registry import ToolRegistry, tool
from tripmate.models import ToolResult
from typing import Annotated

from tests.fakes import FakeLLMClient


@tool
def stub_guide(query: Annotated[str, "topic"]) -> ToolResult:
    """Search the guide."""
    return ToolResult.ok("stub_guide", {"chunks": [
        {"ref": "tokyo/VISA & ENTRY", "city": "tokyo", "section": "VISA & ENTRY",
         "text": "Visa-free for many nationalities.", "score": 0.9}
    ]})


def _client(script) -> TestClient:
    registry = ToolRegistry()
    registry.register(stub_guide)
    agent = Agent(llm=FakeLLMClient(script), registry=registry)
    return TestClient(create_app(agent=agent))


def test_health_reports_ok():
    response = _client([]).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_registered_tools():
    body = _client([]).get("/health").json()
    assert body["tools"] == ["stub_guide"]


def test_chat_returns_the_answer():
    client = _client([LLMResponse(content="Hello from TripMate.")])
    body = client.post("/chat", json={"query": "hello"}).json()

    assert body["answer"] == "Hello from TripMate."


def test_chat_returns_citations_and_trace():
    client = _client([
        LLMResponse(tool_calls=[ToolCall(id="1", name="stub_guide",
                                         arguments={"query": "visa"})]),
        LLMResponse(content="Visa-free [tokyo/VISA & ENTRY]."),
    ])
    body = client.post("/chat", json={"query": "visa for Japan?"}).json()

    assert body["citations"] == ["tokyo/VISA & ENTRY"]
    assert len(body["trace"]) > 0


def test_chat_echoes_the_session_id():
    client = _client([LLMResponse(content="hi")])
    body = client.post("/chat", json={"query": "hi", "session_id": "abc123"}).json()

    assert body["session_id"] == "abc123"


def test_chat_reports_cost_and_latency():
    client = _client([LLMResponse(content="hi", prompt_tokens=10,
                                  completion_tokens=2, cost_usd=0.0001)])
    body = client.post("/chat", json={"query": "hi"}).json()

    assert body["prompt_tokens"] == 10
    assert body["latency_ms"] >= 0


def test_missing_query_field_is_rejected_with_422():
    assert _client([]).post("/chat", json={}).status_code == 422


def test_empty_query_string_is_rejected_with_422():
    assert _client([]).post("/chat", json={"query": ""}).status_code == 422


def test_sessions_endpoint_returns_404_without_a_store():
    assert _client([]).get("/sessions/nope").status_code == 404
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tripmate.adapters.api'`

- [ ] **Step 3: Implement `src/tripmate/adapters/api.py`**

```python
"""FastAPI adapter. A thin shell over the same Agent the CLI uses."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tripmate.core.agent import Agent
from tripmate.db import SessionStore

API_TITLE = "TripMate"
API_VERSION = "0.1.0"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    trace: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    session_id: str
    was_cached: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    tools: list[str]
    chunk_count: int | None = None


def create_app(agent: Agent | None = None, store: SessionStore | None = None) -> FastAPI:
    app = FastAPI(title=API_TITLE, version=API_VERSION)

    def _agent() -> Agent:
        if agent is None:
            from tripmate.bootstrap import build_agent
            return build_agent()
        return agent

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        active = _agent()
        chunk_count: int | None
        try:
            from tripmate.rag.store import build_store
            chunk_count = build_store().count()
        except Exception:
            chunk_count = None
        return HealthResponse(
            status="ok",
            model=getattr(active._settings, "llm_model", "unknown"),
            tools=active._registry.names(),
            chunk_count=chunk_count,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        result = _agent().chat(request.query, session_id=request.session_id)
        return ChatResponse(
            answer=result.answer,
            citations=[c.ref for c in result.citations],
            trace=[event.model_dump() for event in result.trace],
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            session_id=result.session_id,
            was_cached=result.was_cached,
        )

    @app.get("/sessions/{session_id}")
    def session_history(session_id: str) -> dict[str, Any]:
        if store is None:
            raise HTTPException(status_code=404, detail="session history is not enabled")
        history = store.history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "turns": history}

    return app


app = create_app()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`
Expected: 9 passed

- [ ] **Step 5: Manually verify the running API**

```bash
uv run uvicorn tripmate.adapters.api:app --port 8000 &
sleep 5
curl -s localhost:8000/health | python3 -m json.tool
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"query": "What should I pack for Tokyo in December?"}' | python3 -m json.tool
kill %1
```

Expected: health reports both tools and `chunk_count: 20`; the chat response carries an answer, citations, and a trace containing two `TOOL_CALL` events. Interactive docs are at `http://localhost:8000/docs`.

- [ ] **Step 6: Commit**

```bash
git add src/tripmate/adapters/api.py tests/test_api.py
git commit -m "feat: fastapi adapter exposing chat, health and session history"
```

---

### Task 18: Eval harness — dataset and deterministic scoring

**Files:**
- Create: `evals/__init__.py`, `evals/dataset.yaml`, `evals/deterministic.py`, `evals/run_evals.py`
- Test: `tests/test_evals_deterministic.py`

**Interfaces:**
- Consumes: `Agent` (Task 12), `build_agent` (Task 16)
- Produces:
  - `EvalCase(id, query, expected_tools, must_mention, must_not_mention, must_cite, must_refuse, session_id)`
  - `load_dataset(path) -> list[EvalCase]`
  - `score_case(case, response) -> CaseScore` with `tool_selection_correct`, `citations_valid`, `refusal_correct`, `mentions_ok`, `passed`
  - `run_deterministic(agent, cases) -> DeterministicReport` with `accuracy`, `citation_rate`, `refusal_accuracy`, `p50_latency_ms`, `p95_latency_ms`, `total_cost_usd`, `rows`

Layer 1 uses **no LLM judging** — it is exact assertion, so it runs fast, free, and in CI. This is the layer most candidates omit entirely.

- [ ] **Step 1: Write `evals/dataset.yaml`**

```yaml
# TripMate evaluation dataset.
# expected_tools: the exact set the agent should call (order-insensitive).
- id: rag_visa_tokyo
  query: "Do I need a visa to visit Japan as a tourist?"
  expected_tools: [search_destination_guide]
  must_mention: ["visa"]
  must_cite: ["tokyo/VISA & ENTRY"]

- id: rag_customs_tokyo
  query: "What local customs should I know about in Tokyo?"
  expected_tools: [search_destination_guide]
  must_mention: ["bow", "tip"]

- id: rag_safety_bangkok
  query: "Is Bangkok safe for tourists?"
  expected_tools: [search_destination_guide]

- id: rag_best_time_barcelona
  query: "When is the best time of year to visit Barcelona?"
  expected_tools: [search_destination_guide]

- id: rag_customs_barcelona
  query: "Any etiquette tips for Barcelona?"
  expected_tools: [search_destination_guide]

- id: rag_safety_reykjavik
  query: "What safety issues should I know about in Reykjavik?"
  expected_tools: [search_destination_guide]

- id: rag_visa_bangkok
  query: "What are the entry requirements for Thailand?"
  expected_tools: [search_destination_guide]

- id: weather_reykjavik_january
  query: "How cold does Reykjavik get in January?"
  expected_tools: [get_weather_forecast]

- id: weather_bangkok_july
  query: "What's the weather like in Bangkok in July?"
  expected_tools: [get_weather_forecast]

- id: weather_barcelona_august
  query: "How hot is Barcelona in August?"
  expected_tools: [get_weather_forecast]

- id: weather_tokyo_april
  query: "What are typical temperatures in Tokyo in April?"
  expected_tools: [get_weather_forecast]

- id: multi_packing_tokyo_december
  query: "What should I pack for Tokyo in December?"
  expected_tools: [search_destination_guide, get_weather_forecast]
  must_mention: ["layer"]
  must_cite: ["tokyo/PACKING TIPS"]

- id: multi_packing_reykjavik_january
  query: "I'm going to Reykjavik in January. What should I bring?"
  expected_tools: [search_destination_guide, get_weather_forecast]

- id: multi_packing_bangkok_july
  query: "What clothes do I need for Bangkok in July?"
  expected_tools: [search_destination_guide, get_weather_forecast]

- id: multi_packing_barcelona_august
  query: "Packing advice for Barcelona in August please."
  expected_tools: [search_destination_guide, get_weather_forecast]

- id: multi_trip_planning_tokyo
  query: "I'm visiting Tokyo in December. What's the weather and what should I know before I go?"
  expected_tools: [search_destination_guide, get_weather_forecast]

- id: none_greeting
  query: "Hi there!"
  expected_tools: []

- id: none_capabilities
  query: "What can you help me with?"
  expected_tools: []
  must_mention: ["tokyo", "bangkok"]

- id: none_thanks
  query: "Thanks, that's really helpful."
  expected_tools: []

- id: scope_book_flight
  query: "Can you book my flight to Barcelona?"
  expected_tools: []
  must_refuse: true
  must_not_mention: ["booked", "confirmed", "reservation number", "i have booked"]

- id: scope_book_hotel
  query: "Please reserve a hotel in Tokyo for me next week."
  expected_tools: []
  must_refuse: true
  must_not_mention: ["reserved", "confirmed", "booking reference"]

- id: scope_cancel_trip
  query: "Cancel my trip and refund my payment."
  expected_tools: []
  must_refuse: true
  must_not_mention: ["cancelled", "refunded"]

- id: scope_unrelated_code
  query: "Write me a Python script to sort a list of dictionaries."
  expected_tools: []
  must_refuse: true

- id: scope_unrelated_stocks
  query: "Should I buy Tesla stock right now?"
  expected_tools: []
  must_refuse: true

- id: error_unknown_city_guide
  query: "What are the visa requirements for Paris?"
  expected_tools: [search_destination_guide]
  must_mention: ["tokyo", "bangkok"]
  must_not_mention: ["schengen visa is required for paris"]

- id: error_unknown_city_weather
  query: "What's the weather in Atlantis in July?"
  expected_tools: [get_weather_forecast]

- id: error_ambiguous_packing
  query: "What should I pack?"
  expected_tools: []
  must_mention: ["?"]

- id: error_ambiguous_when
  query: "When should I go?"
  expected_tools: []
  must_mention: ["?"]

- id: grounding_no_invention_tokyo
  query: "How much does the Tokyo metro day pass cost?"
  expected_tools: [search_destination_guide]
  must_not_mention: ["yen per day", "costs exactly"]

- id: grounding_out_of_corpus_city
  query: "Tell me about local customs in Seoul."
  expected_tools: [search_destination_guide]
  must_mention: ["tokyo", "reykjavik"]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_evals_deterministic.py
from pathlib import Path

from evals.deterministic import EvalCase, load_dataset, score_case
from tripmate.core.trace import EventType
from tripmate.models import AgentResponse, Citation, TraceEvent


def _response(answer: str, tools: list[str], citations: list[str] | None = None):
    trace = [
        TraceEvent(seq=index + 1, event_type=EventType.TOOL_CALL, timestamp=0.0,
                   payload={"tool": tool})
        for index, tool in enumerate(tools)
    ]
    return AgentResponse(
        answer=answer,
        citations=[Citation(city=c.split("/")[0], section=c.split("/")[1])
                   for c in (citations or [])],
        trace=trace,
    )


def test_dataset_loads_every_case():
    cases = load_dataset(Path("evals/dataset.yaml"))
    assert len(cases) >= 30


def test_dataset_case_ids_are_unique():
    cases = load_dataset(Path("evals/dataset.yaml"))
    assert len({case.id for case in cases}) == len(cases)


def test_correct_tool_selection_scores_true():
    case = EvalCase(id="c", query="q", expected_tools=["get_weather_forecast"])
    score = score_case(case, _response("Cold.", ["get_weather_forecast"]))
    assert score.tool_selection_correct


def test_tool_selection_ignores_call_order():
    case = EvalCase(id="c", query="q",
                    expected_tools=["get_weather_forecast", "search_destination_guide"])
    score = score_case(case, _response(
        "x", ["search_destination_guide", "get_weather_forecast"]))
    assert score.tool_selection_correct


def test_missing_expected_tool_scores_false():
    case = EvalCase(id="c", query="q",
                    expected_tools=["search_destination_guide", "get_weather_forecast"])
    score = score_case(case, _response("x", ["search_destination_guide"]))
    assert not score.tool_selection_correct


def test_required_mention_is_case_insensitive():
    case = EvalCase(id="c", query="q", expected_tools=[], must_mention=["LAYER"])
    assert score_case(case, _response("Pack layers.", [])).mentions_ok


def test_forbidden_mention_fails_the_case():
    case = EvalCase(id="c", query="q", expected_tools=[],
                    must_not_mention=["booked"])
    assert not score_case(case, _response("I have booked your flight.", [])).mentions_ok


def test_refusal_is_detected_from_refusal_phrases():
    case = EvalCase(id="c", query="q", expected_tools=[], must_refuse=True)
    assert score_case(case, _response("I can't book flights.", [])).refusal_correct


def test_missing_refusal_scores_false():
    case = EvalCase(id="c", query="q", expected_tools=[], must_refuse=True)
    assert not score_case(case, _response("Sure, done!", [])).refusal_correct


def test_required_citation_present_scores_true():
    case = EvalCase(id="c", query="q", expected_tools=[],
                    must_cite=["tokyo/PACKING TIPS"])
    score = score_case(case, _response("Layers.", [], ["tokyo/PACKING TIPS"]))
    assert score.citations_valid


def test_required_citation_absent_scores_false():
    case = EvalCase(id="c", query="q", expected_tools=[],
                    must_cite=["tokyo/PACKING TIPS"])
    assert not score_case(case, _response("Layers.", [], [])).citations_valid


def test_case_passes_only_when_every_check_passes():
    case = EvalCase(id="c", query="q", expected_tools=["get_weather_forecast"],
                    must_mention=["cold"])
    assert score_case(case, _response("It is cold.", ["get_weather_forecast"])).passed
    assert not score_case(case, _response("It is warm.", ["get_weather_forecast"])).passed
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_evals_deterministic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.deterministic'`

- [ ] **Step 4: Implement `evals/deterministic.py`**

```python
"""Layer 1: deterministic scoring. No LLM judging — exact assertions only."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tripmate.core.trace import EventType
from tripmate.models import AgentResponse

REFUSAL_PHRASES = (
    "can't", "cannot", "unable to", "not able to", "outside what i",
    "i don't have the ability", "i'm not able",
)


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_tools: list[str]
    must_mention: list[str] = field(default_factory=list)
    must_not_mention: list[str] = field(default_factory=list)
    must_cite: list[str] = field(default_factory=list)
    must_refuse: bool = False
    session_id: str | None = None


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    tool_selection_correct: bool
    citations_valid: bool
    refusal_correct: bool
    mentions_ok: bool
    latency_ms: float
    cost_usd: float
    tools_called: list[str]
    answer: str

    @property
    def passed(self) -> bool:
        return (
            self.tool_selection_correct
            and self.citations_valid
            and self.refusal_correct
            and self.mentions_ok
        )


@dataclass(frozen=True)
class DeterministicReport:
    rows: list[CaseScore]

    @property
    def accuracy(self) -> float:
        return self._ratio([row.tool_selection_correct for row in self.rows])

    @property
    def citation_rate(self) -> float:
        return self._ratio([row.citations_valid for row in self.rows])

    @property
    def refusal_accuracy(self) -> float:
        return self._ratio([row.refusal_correct for row in self.rows])

    @property
    def pass_rate(self) -> float:
        return self._ratio([row.passed for row in self.rows])

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(95)

    @property
    def total_cost_usd(self) -> float:
        return sum(row.cost_usd for row in self.rows)

    def _ratio(self, flags: list[bool]) -> float:
        return sum(flags) / len(flags) if flags else 0.0

    def _percentile(self, percentile: int) -> float:
        values = sorted(row.latency_ms for row in self.rows)
        if not values:
            return 0.0
        if percentile == 50:
            return statistics.median(values)
        index = min(int(len(values) * percentile / 100), len(values) - 1)
        return values[index]


def load_dataset(path: str | Path) -> list[EvalCase]:
    raw: list[dict[str, Any]] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=entry["id"],
            query=entry["query"],
            expected_tools=list(entry.get("expected_tools", [])),
            must_mention=list(entry.get("must_mention", [])),
            must_not_mention=list(entry.get("must_not_mention", [])),
            must_cite=list(entry.get("must_cite", [])),
            must_refuse=bool(entry.get("must_refuse", False)),
            session_id=entry.get("session_id"),
        )
        for entry in raw
    ]


def tools_called(response: AgentResponse) -> list[str]:
    return [
        event.payload["tool"]
        for event in response.trace
        if event.event_type == EventType.TOOL_CALL
    ]


def score_case(case: EvalCase, response: AgentResponse) -> CaseScore:
    called = tools_called(response)
    answer_lower = (response.answer or "").lower()
    cited = {citation.ref for citation in response.citations}

    mentions_ok = all(
        phrase.lower() in answer_lower for phrase in case.must_mention
    ) and not any(phrase.lower() in answer_lower for phrase in case.must_not_mention)

    refusal_correct = True
    if case.must_refuse:
        refusal_correct = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)

    return CaseScore(
        case_id=case.id,
        tool_selection_correct=set(called) == set(case.expected_tools),
        citations_valid=all(ref in cited for ref in case.must_cite),
        refusal_correct=refusal_correct,
        mentions_ok=mentions_ok,
        latency_ms=response.latency_ms,
        cost_usd=response.cost_usd,
        tools_called=called,
        answer=response.answer,
    )


def run_deterministic(agent: Any, cases: list[EvalCase]) -> DeterministicReport:
    rows = [
        score_case(case, agent.chat(case.query, session_id=case.session_id))
        for case in cases
    ]
    return DeterministicReport(rows=rows)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_evals_deterministic.py -v`
Expected: 12 passed

- [ ] **Step 6: Implement `evals/run_evals.py`**

```python
"""Eval entry point. Runs the deterministic layer by default; --ragas and
--simulation add the LLM-judged layers."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.deterministic import DeterministicReport, load_dataset, run_deterministic
from tripmate.bootstrap import build_agent

DATASET_PATH = Path(__file__).parent / "dataset.yaml"


def _print_report(console: Console, report: DeterministicReport) -> None:
    table = Table(title="deterministic evaluation")
    table.add_column("case", overflow="fold")
    table.add_column("tools", justify="center", width=6)
    table.add_column("cite", justify="center", width=5)
    table.add_column("refuse", justify="center", width=7)
    table.add_column("text", justify="center", width=5)
    table.add_column("ms", justify="right", width=7)
    table.add_column("called", overflow="fold")

    def mark(flag: bool) -> str:
        return "[green]PASS[/]" if flag else "[red]FAIL[/]"

    for row in report.rows:
        table.add_row(
            row.case_id, mark(row.tool_selection_correct), mark(row.citations_valid),
            mark(row.refusal_correct), mark(row.mentions_ok),
            f"{row.latency_ms:.0f}", ", ".join(row.tools_called) or "-",
        )
    console.print(table)

    summary = Table(title="summary", show_header=False)
    summary.add_row("tool-selection accuracy", f"{report.accuracy:.1%}")
    summary.add_row("citation validity", f"{report.citation_rate:.1%}")
    summary.add_row("refusal accuracy", f"{report.refusal_accuracy:.1%}")
    summary.add_row("overall pass rate", f"{report.pass_rate:.1%}")
    summary.add_row("p50 latency", f"{report.p50_latency_ms:.0f} ms")
    summary.add_row("p95 latency", f"{report.p95_latency_ms:.0f} ms")
    summary.add_row("total cost", f"${report.total_cost_usd:.4f}")
    console.print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TripMate eval suite.")
    parser.add_argument("--ragas", action="store_true", help="add RAGAS RAG metrics")
    parser.add_argument("--simulation", action="store_true",
                        help="add simulated-user conversations")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N cases")
    args = parser.parse_args()

    console = Console()
    agent = build_agent()
    cases = load_dataset(DATASET_PATH)
    if args.limit:
        cases = cases[: args.limit]

    console.print(f"[dim]running {len(cases)} deterministic case(s)...[/dim]")
    _print_report(console, run_deterministic(agent, cases))

    if args.ragas:
        from evals.ragas_eval import run_ragas, print_ragas_report
        print_ragas_report(console, run_ragas(agent, cases))

    if args.simulation:
        from evals.simulation import run_simulations, print_simulation_report
        print_simulation_report(console, run_simulations(agent))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the deterministic evals against a real provider**

```bash
uv run python -m evals.run_evals
```

Expected: a per-case table and a summary. Tool-selection accuracy below ~85% means the system prompt's TOOL POLICY section needs tightening — adjust `core/prompts.py` and re-run. Record the final numbers; they go in the README.

- [ ] **Step 8: Commit**

```bash
git add evals/ tests/test_evals_deterministic.py
git commit -m "feat: eval dataset and deterministic scoring harness"
```

---

### Task 19: Eval harness — RAGAS and simulated user

**Files:**
- Create: `evals/ragas_eval.py`, `evals/simulation.py`
- Test: `tests/test_evals_simulation.py`

**Interfaces:**
- Consumes: `EvalCase` (Task 18), `Agent` (Task 12)
- Produces:
  - `collect_samples(agent, cases) -> list[dict]` — `{question, answer, contexts, reference}` rows for RAGAS
  - `run_ragas(agent, cases) -> dict[str, float]`, `print_ragas_report(console, scores) -> None`
  - `Conversation(id, turns, goal_phrases)`, `CONVERSATIONS: list[Conversation]`
  - `run_simulations(agent) -> list[SimulationResult]`, `print_simulation_report(console, results) -> None`
  - `score_conversation(conversation, answers) -> SimulationResult`

- [ ] **Step 1: Install the eval extras**

```bash
uv pip install -e ".[evals]"
```

- [ ] **Step 2: Implement `evals/ragas_eval.py`**

```python
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

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


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
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    samples = collect_samples(agent, cases)
    if not samples:
        return {}

    result = evaluate(
        Dataset.from_list(samples),
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return {name: float(result[name]) for name in METRIC_NAMES if name in result}


def print_ragas_report(console: Console, scores: dict[str, float]) -> None:
    if not scores:
        console.print("[yellow]RAGAS: no RAG samples collected[/yellow]")
        return
    table = Table(title="RAGAS metrics", show_header=False)
    for name, value in scores.items():
        table.add_row(name.replace("_", " "), f"{value:.3f}")
    console.print(table)
```

> The trace must carry the retrieved text for this to work. Add it in Task 12's
> `_dispatch_all`: when recording `TOOL_RESULT` for `search_destination_guide`, include
> `texts=[chunk["text"] for chunk in (result.data or {}).get("chunks", [])]`.
> Make that edit now and re-run `uv run pytest tests/test_agent.py` to confirm it still passes.

- [ ] **Step 3: Write the failing test for the simulation layer**

```python
# tests/test_evals_simulation.py
from evals.simulation import Conversation, score_conversation


def test_goal_phrases_found_across_turns_score_complete():
    conversation = Conversation(
        id="c1", turns=["going to Tokyo in December", "what should I pack?"],
        goal_phrases=["layer", "cold"],
    )
    result = score_conversation(conversation,
                                ["Great, Tokyo in December.",
                                 "Pack warm layers; it is cold."])
    assert result.goal_completion == 1.0


def test_partially_met_goals_score_between_zero_and_one():
    conversation = Conversation(id="c1", turns=["a", "b"],
                                goal_phrases=["layer", "umbrella"])
    result = score_conversation(conversation, ["ok", "pack layers"])
    assert result.goal_completion == 0.5


def test_context_is_retained_when_a_later_answer_names_the_destination():
    conversation = Conversation(
        id="c1", turns=["I'm going to Bangkok", "what should I pack?"],
        goal_phrases=[], context_phrases=["bangkok"],
    )
    result = score_conversation(conversation,
                                ["Nice choice.", "For Bangkok, pack light fabrics."])
    assert result.context_retained is True


def test_context_is_not_retained_when_later_answers_drop_the_destination():
    conversation = Conversation(
        id="c1", turns=["I'm going to Bangkok", "what should I pack?"],
        goal_phrases=[], context_phrases=["bangkok"],
    )
    result = score_conversation(conversation, ["Nice choice.", "Pack light fabrics."])
    assert result.context_retained is False


def test_a_conversation_without_goals_scores_one():
    conversation = Conversation(id="c1", turns=["hi"], goal_phrases=[])
    assert score_conversation(conversation, ["hello"]).goal_completion == 1.0
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `uv run pytest tests/test_evals_simulation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evals.simulation'`

- [ ] **Step 5: Implement `evals/simulation.py`**

```python
"""Layer 3: simulated multi-turn traveler conversations.

Scripted turns stand in for a real user across a whole conversation, scoring goal
completion and whether context carried between turns. Single-turn evals cannot
catch a follow-up like "and what about Bangkok?" losing the thread.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: list[str]
    goal_phrases: list[str] = field(default_factory=list)
    context_phrases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationResult:
    conversation_id: str
    goal_completion: float
    context_retained: bool
    answers: list[str]

    @property
    def passed(self) -> bool:
        return self.goal_completion >= 1.0 and self.context_retained


CONVERSATIONS: list[Conversation] = [
    Conversation(
        id="tokyo_winter_trip",
        turns=[
            "I'm planning a trip to Tokyo in December.",
            "What should I pack?",
            "And is it safe there?",
        ],
        goal_phrases=["layer", "safe"],
        context_phrases=["tokyo"],
    ),
    Conversation(
        id="switch_destination",
        turns=[
            "What's the weather like in Bangkok in July?",
            "And what about Barcelona instead?",
        ],
        goal_phrases=["barcelona"],
        context_phrases=["barcelona"],
    ),
    Conversation(
        id="scope_then_recover",
        turns=[
            "Can you book me a flight to Reykjavik?",
            "Okay, then just tell me what to pack for January.",
        ],
        goal_phrases=["pack"],
        context_phrases=["reykjavik"],
    ),
]


def score_conversation(
    conversation: Conversation, answers: list[str]
) -> SimulationResult:
    combined = " ".join(answers).lower()

    if conversation.goal_phrases:
        met = sum(1 for phrase in conversation.goal_phrases
                  if phrase.lower() in combined)
        goal_completion = met / len(conversation.goal_phrases)
    else:
        goal_completion = 1.0

    if conversation.context_phrases and len(answers) > 1:
        later = " ".join(answers[1:]).lower()
        context_retained = all(
            phrase.lower() in later for phrase in conversation.context_phrases
        )
    else:
        context_retained = True

    return SimulationResult(
        conversation_id=conversation.id,
        goal_completion=goal_completion,
        context_retained=context_retained,
        answers=answers,
    )


def run_simulations(
    agent: Any, conversations: list[Conversation] | None = None
) -> list[SimulationResult]:
    results: list[SimulationResult] = []
    for conversation in conversations or CONVERSATIONS:
        session_id = uuid.uuid4().hex[:12]
        answers = [
            agent.chat(turn, session_id=session_id).answer
            for turn in conversation.turns
        ]
        results.append(score_conversation(conversation, answers))
    return results


def print_simulation_report(
    console: Console, results: list[SimulationResult]
) -> None:
    table = Table(title="simulated conversations")
    table.add_column("conversation")
    table.add_column("goal", justify="right", width=7)
    table.add_column("context", justify="center", width=9)
    table.add_column("result", justify="center", width=7)

    for result in results:
        table.add_row(
            result.conversation_id,
            f"{result.goal_completion:.0%}",
            "[green]kept[/]" if result.context_retained else "[red]lost[/]",
            "[green]PASS[/]" if result.passed else "[red]FAIL[/]",
        )
    console.print(table)
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/test_evals_simulation.py -v`
Expected: 5 passed

- [ ] **Step 7: Run the full eval suite against a real provider**

```bash
uv run python -m evals.run_evals --ragas --simulation
```

Expected: the deterministic table, RAGAS metrics, and the simulation table. Record every number for the README.

- [ ] **Step 8: Commit**

```bash
git add evals/ragas_eval.py evals/simulation.py tests/test_evals_simulation.py
git commit -m "feat: ragas metrics and simulated multi-turn conversation evals"
```

---

### Task 20: Packaging, documentation and captured traces

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docs/architecture.md`, `README.md`, `traces/example_*.jsonl`
- Modify: `pyproject.toml` (coverage gate)

**Interfaces:**
- Consumes: everything
- Produces: the submission artifacts

- [ ] **Step 1: Write the multi-stage `Dockerfile`**

```dockerfile
# --- build stage: resolve and install dependencies into a virtualenv ---
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# --- runtime stage: copy only the virtualenv and the application ---
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN useradd --create-home --uid 1000 tripmate
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY data/ ./data/

USER tripmate
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "tripmate.adapters.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
cat > .dockerignore <<'EOF'
.venv/
.git/
.chroma/
.weather_cache/
*.db
traces/
htmlcov/
.pytest_cache/
__pycache__/
.env
EOF
```

- [ ] **Step 2: Build and run the container**

```bash
docker build -t tripmate:local .
docker run --rm -p 8000:8000 -e LLM_MODEL=gpt-4o-mini -e LLM_API_KEY=$LLM_API_KEY tripmate:local &
sleep 20
curl -s localhost:8000/health | python3 -m json.tool
docker stop $(docker ps -q --filter ancestor=tripmate:local)
```

Expected: `{"status": "ok", "tools": [...], "chunk_count": 20}`

- [ ] **Step 3: Write `docs/architecture.md`**

Copy the mermaid diagram and the request-flow block verbatim from spec section 3
(`docs/superpowers/specs/2026-09-16-tripmate-agent-design.md`), and add one short
paragraph per component naming the file it lives in. GitHub renders mermaid inline, so
no image export is needed.

- [ ] **Step 4: Capture example traces as deliverables**

```bash
rm -f traces/*.jsonl
uv run python - <<'PY'
import json, shutil
from pathlib import Path
from tripmate.bootstrap import build_agent

CASES = {
    "example_single_tool_rag":   "Do I need a visa to visit Japan as a tourist?",
    "example_single_tool_weather": "How cold does Reykjavik get in January?",
    "example_multi_tool_packing": "What should I pack for Tokyo in December?",
    "example_out_of_scope":       "Can you book my flight to Barcelona?",
}

agent = build_agent()
for name, query in CASES.items():
    response = agent.chat(query)
    source = Path("traces") / f"{response.session_id}.jsonl"
    shutil.move(source, Path("traces") / f"{name}.jsonl")
    print(f"{name}: {len(response.trace)} events -> {response.answer[:90]}")
PY
ls traces/
```

Then capture the failure path with networking disabled:

```bash
# In one shell, block outbound DNS for the weather host, or simply run offline.
uv run python - <<'PY'
import shutil
from pathlib import Path
from unittest.mock import patch
import httpx
from tripmate.bootstrap import build_agent

with patch("httpx.get", side_effect=httpx.TimeoutException("simulated outage")):
    agent = build_agent()
    response = agent.chat("What should I pack for Tokyo in December?")
    shutil.move(Path("traces") / f"{response.session_id}.jsonl",
                Path("traces") / "example_weather_fallback.jsonl")
    print(response.answer)
PY
```

Expected: `example_weather_fallback.jsonl` contains a `FALLBACK_USED` event. Confirm with:

```bash
grep -c FALLBACK_USED traces/example_weather_fallback.jsonl
```

- [ ] **Step 5: Verify coverage meets the floor**

```bash
uv run pytest --cov --cov-report=term-missing
```

Expected: total coverage 80% or higher. If it is short, add tests for the uncovered
branches the report names — do not lower the threshold.

Then pin the gate:

```toml
# append to pyproject.toml
[tool.coverage.report]
fail_under = 80
show_missing = true
```

- [ ] **Step 6: Write `README.md`**

Required sections, in this order. Fill every number from the runs above — no placeholders.

1. **What this is** — one paragraph, plus the mermaid architecture diagram.
2. **Quickstart**
   ```bash
   git clone <repo> && cd tripmate
   uv venv && uv pip install -e ".[dev]"
   cp .env.example .env        # set LLM_MODEL and LLM_API_KEY
   uv run tripmate-ingest      # loads 20 chunks into ChromaDB
   uv run tripmate             # CLI
   uv run uvicorn tripmate.adapters.api:app --reload   # API at :8000/docs
   docker build -t tripmate . && docker run -p 8000:8000 --env-file .env tripmate
   ```
   State plainly that `LLM_MODEL=ollama/qwen3.5` runs the whole system with **no API key**.
3. **Architecture** — link `docs/architecture.md`, describe the request flow in prose.
4. **Tool schemas as given to the LLM** — paste the real output of:
   ```bash
   uv run python -c "
   import json
   from tripmate.tools.registry import ToolRegistry
   from tripmate.tools.destination import search_destination_guide
   from tripmate.tools.weather import get_weather_forecast
   r = ToolRegistry(); r.register(search_destination_guide); r.register(get_weather_forecast)
   print(json.dumps(r.schemas(), indent=2))"
   ```
5. **Example runs** — five, each showing input, the trace, and the answer, taken from
   `traces/`: single-tool RAG, single-tool weather, **multi-tool packing**,
   **out-of-scope booking**, and the **weather fallback**.
6. **Design decisions** — spec section 2's table, including *Why not LangGraph* verbatim.
7. **Evaluation** — the three layers, with the real numbers from Tasks 18 and 19, plus
   how to reproduce: `uv run python -m evals.run_evals --ragas --simulation`.
8. **Testing** — what each test file covers, the coverage percentage, and `uv run pytest`.
9. **Scalability** — spec section 16, all four questions.
10. **Known limitations** — spec section 17.
11. **Future improvements** — spec section 18.
12. **Deliberately out of scope** — spec section 15's table, with reasons.

- [ ] **Step 7: Final verification**

```bash
uv run pytest --cov -q
uv run python -m evals.run_evals --limit 5
ls traces/
docker build -t tripmate:final . >/dev/null && echo "docker build ok"
```

Expected: all tests pass, coverage at or above 80%, five example traces present, image builds.

- [ ] **Step 8: Commit**

```bash
git add Dockerfile .dockerignore README.md docs/architecture.md traces/ pyproject.toml
git commit -m "docs: readme, architecture diagram, dockerfile and captured example traces"
```

- [ ] **Step 9: Record the Loom video (5–8 minutes)**

| Minutes | Segment |
|---|---|
| 0:00–0:45 | **Overview** — the problem, and the approach in one sentence |
| 0:45–2:00 | **Architecture** — walk the mermaid diagram, name each component's file |
| 2:00–4:00 | **Demo** — single-tool, multi-tool packing (both tools visible in the trace), out-of-scope refusal, then **kill the network and show `FALLBACK_USED` engage** |
| 4:00–5:30 | **Code** — `tools/registry.py` schema derivation, then `core/agent.py` loop |
| 5:30–6:45 | **Decisions** — LiteLLM for provider independence, raw loop over LangGraph (and when you would switch), Chroma for metadata pre-filtering, climate-normal over forecast. Then **swap `LLM_MODEL` to Ollama and rerun the same query live** |
| 6:45–8:00 | **Evals** — show the scorecard; close on limitations and future work |

The induced failure and the live provider swap are the two segments almost no submission has. Do not cut them for time.

---

## Self-Review

**Spec coverage.** Every numbered spec section maps to at least one task:

| Spec section | Tasks |
|---|---|
| §2 D1 LiteLLM / D7 FastAPI / D8 Docker / D9 persistence | 8, 17, 20, 10 |
| §2 D2 raw loop, "why not LangGraph" | 12, 20 (README §6) |
| §2 D3/D4 Chroma and fastembed | 5 |
| §2 D5 weather dual path | 7 |
| §3 architecture and request flow | 12, 20 |
| §4 module structure | 1–17 |
| §5.1 registry | 3 |
| §5.2 vector store | 5 |
| §5.3 destination tool | 6 |
| §5.4 weather tool | 7 |
| §5.5 agent core | 12 |
| §5.6 semantic cache | 11 |
| §5.7 citations | 12 |
| §6 error handling | 6, 7, 12, 14 |
| §7 configuration | 1 |
| §8 logging and tracing | 9 |
| §9 persistence | 10 |
| §10 adapters | 16, 17 |
| §11 system prompt | 12 |
| §12 testing | 3–17 |
| §13 eval harness | 18, 19 |
| §14 deliverables | 20 |
| §15–18 out of scope, scalability, limitations, future work | 20 (README) |
| §19 dependencies | 1 |

**Deviations from the spec, deliberate:**
1. **`bootstrap.py` added** (Task 16) — not named in the spec's module table. Without it, the CLI and API would each wire the agent separately and drift. One wiring point, ~30 lines.
2. **`structlog` not wired in Task 9.** The `Tracer` provides the structured events the spec requires, and `rich` renders them. Adding `structlog` on top would be a second logging system doing the same job. It stays in `pyproject.toml` for application-level logs; if no such need appears by Task 20, remove the dependency rather than leave it unused.

**Type consistency, checked across tasks:**
- `ToolResult.ok/no_data/error` — signatures identical in Tasks 2, 6, 7, 12.
- `Chunk.ref` returns `"{city}/{section}"`; `Citation.ref` returns the same shape. Task 12's `validate_citations` compares `Citation.ref` against refs gathered from tool data `chunks[].ref`, which Task 6 populates from `Chunk.ref`. Consistent.
- `VectorStore.search(query, k, city, min_score)` — same parameter names in Tasks 5, 6.
- `Agent.chat(query, session_id)` — same in Tasks 12, 13, 16, 17, 18, 19.
- `EventType.*` constants — defined in Task 9, used unchanged in 12, 13, 14, 15, 16, 18, 19.
- `FakeLLMClient.complete(messages, tools)` matches `LLMClient.complete` and the `SupportsComplete` Protocol in Task 12.
- `CITATION_RE` is defined once in `models.py` (Task 2) and imported by `agent.py` (Task 12) — not redefined.

**One forward dependency, flagged:** Task 19's RAGAS layer needs retrieved chunk text on `TOOL_RESULT` events. Task 19 Step 2 contains the exact edit to `core/agent.py` and instructs re-running Task 12's tests. If Tasks 12 and 19 are executed by different workers, this is the one place they touch the same file.
