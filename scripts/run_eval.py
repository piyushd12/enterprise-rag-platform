"""
CLI trigger for the RAGAS evaluation harness.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --sample-size 5
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_app.evaluation.datasets import load_qa_dataset
from rag_app.evaluation.harness import print_summary, run_ragas_eval
from rag_app.observability import ObservabilityProvider, configure_langsmith, configure_logfire


async def main(sample_size: int | None) -> None:
    configure_logfire()
    configure_langsmith()
    obs = ObservabilityProvider()

    qa_items = load_qa_dataset()
    if sample_size is not None:
        qa_items = qa_items[:sample_size]

    print(f"Running RAGAS evaluation on {len(qa_items)} question(s)...")
    result = await run_ragas_eval(qa_items=qa_items, obs=obs)
    print_summary(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAGAS evaluation harness.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate only the first N questions (default: all).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.sample_size))
