"""
Seed the vector store with sample documents.

Usage:
    uv run python scripts/seed_vectorstore.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_app.config.settings import settings
from rag_app.embeddings.fastembed_provider import FastEmbedProvider
from rag_app.ingestion.pipeline import ingest_file
from rag_app.observability import configure_logfire, configure_langsmith, ObservabilityProvider
from rag_app.vectorstore.qdrant_store import QdrantStore


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


async def main() -> None:
    # Bootstrap observability
    configure_logfire()
    configure_langsmith()
    obs = ObservabilityProvider()

    # Initialize providers
    embedder = FastEmbedProvider()
    store = QdrantStore(obs=obs, embedding_dimension=embedder.dimension())

    # Ensure collection exists
    await store.ensure_collection()

    # Find all sample documents
    sample_files = list(SAMPLE_DIR.glob("*.md")) + list(SAMPLE_DIR.glob("*.txt"))
    if not sample_files:
        print(f"No sample documents found in {SAMPLE_DIR}")
        return

    print(f"Found {len(sample_files)} sample document(s)")

    for file_path in sample_files:
        print(f"\nIngesting: {file_path.name}")
        result = await ingest_file(
            file_path=file_path,
            embedding_provider=embedder,
            vector_store=store,
            obs=obs,
        )
        print(f"  ✓ source_id={result['source_id']}, chunks={result['chunk_count']}")

    print("\nSeeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
