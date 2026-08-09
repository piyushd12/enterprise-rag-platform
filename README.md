# RAG Application

Enterprise-grade Retrieval-Augmented Generation application demonstrating production system-design: caching, async queues, dual observability (LangSmith + Logfire), evaluation-driven iteration, and clean hexagonal architecture.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangChain + LangGraph |
| LLM | Groq (primary) → OpenRouter Nemotron (fallback) |
| Embeddings | FastEmbed (local, no API key) |
| Vector DB | Qdrant |
| Cache + Broker | Redis |
| Task Queue | Celery |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM Observability | LangSmith |
| Infra Observability | Logfire |
| Evaluation | RAGAS |

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker & Docker Compose

### Setup

```bash
# 1. Clone and enter the project
cd Rag-application

# 2. Copy environment template and fill in API keys
cp .env.example .env
# Edit .env with your actual keys

# 3. Create virtual environment and install dependencies
uv venv
uv sync

# 4. Start infrastructure (Qdrant + Redis)
make infra
# or: docker compose up -d

# 5. Seed sample documents
make seed
# or: uv run python scripts/seed_vectorstore.py

# 6. Start the FastAPI backend
make dev
# or: uv run uvicorn rag_app.main:app --reload --host 0.0.0.0 --port 8000

# 7. (In another terminal) Start the Streamlit UI
make ui
# or: uv run streamlit run ui/app.py --server.port 8501
```

### Running the Celery Worker (for async ingestion)

```bash
make worker
# or: uv run celery -A rag_app.queue.celery_app worker --loglevel=info
```

## Architecture

See the architecture document for full details on:
- Hexagonal/layered design with abstract interfaces
- LLM fallback router (Groq → OpenRouter)
- LangGraph state graph with extensible node design
- Dual observability: LangSmith (LLM pipeline) + Logfire (infrastructure)
- Redis cache-aside pattern with TTL and invalidation
- Celery async ingestion with retry/backoff

## Environment Variables

See `.env.example` for all required and optional variables.

## uv Workflow

This project uses `uv` exclusively for package management:

```bash
uv add <package>       # Add a dependency
uv sync                # Install all dependencies
uv run <command>       # Run within the venv
uv venv                # Create virtual environment
```

Never use `pip install` directly.
