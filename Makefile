.PHONY: infra dev worker ui seed eval lint test

# Start infrastructure (Qdrant + Redis)
infra:
	docker compose up -d

# Stop infrastructure
infra-down:
	docker compose down

# Run FastAPI dev server
dev:
	uv run uvicorn rag_app.main:app --reload --reload-dir src/rag_app --host 0.0.0.0 --port 8000

# Run Celery worker
worker:
	uv run celery -A rag_app.queue.celery_app worker --loglevel=info

# Run Streamlit UI
ui:
	uv run streamlit run ui/app.py --server.port 8501

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
