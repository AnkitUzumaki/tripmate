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
