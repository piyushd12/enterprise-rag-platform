.PHONY: up down dev worker seed eval lint test

# Start the full stack (Qdrant, Redis, Celery worker, FastAPI + chat UI)
up:
	docker compose up -d --build

# Stop the full stack
down:
	docker compose down

# Run FastAPI dev server (serves the chat UI at http://localhost:8000)
dev:
	uv run uvicorn rag_app.main:app --reload --reload-dir src/rag_app --host 0.0.0.0 --port 8000

# Run Celery worker
worker:
	uv run celery -A rag_app.queue.celery_app worker --loglevel=info

# Seed vector store with sample documents
seed:
	uv run python scripts/seed_vectorstore.py

# Run RAGAS evaluation
eval:
	uv run python scripts/run_eval.py

# Run tests
test:
	uv run pytest tests/ -v

# Lint
lint:
	uv run ruff check src/ tests/
