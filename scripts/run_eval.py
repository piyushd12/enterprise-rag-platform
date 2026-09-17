"""
CLI trigger for the RAGAS evaluation harness.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --sample-size 5
    uv run python scripts/run_eval.py --use-hyde
    uv run python scripts/run_eval.py --use-reranker
    uv run python scripts/run_eval.py --use-reranker --use-bm25
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_app.evaluation.datasets import load_qa_dataset
from rag_app.evaluation.harness import print_summary, run_ragas_eval
from rag_app.observability import ObservabilityProvider, configure_langsmith, configure_logfire


async def main(
    sample_size: int | None, use_hyde: bool, use_reranker: bool, use_bm25: bool
) -> None:
    configure_logfire()
    configure_langsmith()
    obs = ObservabilityProvider()

    qa_items = load_qa_dataset()
    if sample_size is not None:
        qa_items = qa_items[:sample_size]

    print(
        f"Running RAGAS evaluation on {len(qa_items)} question(s) "
        f"(use_hyde={use_hyde}, use_reranker={use_reranker}, use_bm25={use_bm25})..."
    )
    result = await run_ragas_eval(
        qa_items=qa_items,
        obs=obs,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        use_bm25=use_bm25,
    )
    print_summary(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAGAS evaluation harness.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate only the first N questions (default: all).",
    )
    parser.add_argument(
        "--use-hyde",
        action="store_true",
        help="Enable the HyDE query-expansion node before retrieval.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable the cross-encoder reranker after retrieval.",
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        help="Merge BM25 keyword search into the candidate pool before reranking.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.sample_size, args.use_hyde, args.use_reranker, args.use_bm25))
