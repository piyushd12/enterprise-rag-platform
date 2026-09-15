"""
Test dataset management for RAG evaluation.

Loads the hand-authored Q&A ground-truth set used by the RAGAS harness.
Only a reference answer is needed per question -- RAGAS compares the
live /chat pipeline's actual retrieval + generation against it, so no
hand-picked context chunks are stored here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "eval" / "qa_dataset.json"
)


class QAItem(TypedDict):
    """A single ground-truth question/answer pair."""

    id: str
    source_doc: str
    question: str
    ground_truth: str


def load_qa_dataset(path: Path | None = None) -> list[QAItem]:
    """Load the Q&A ground-truth set from disk.

    Args:
        path: Override path to the dataset JSON file. Defaults to
            data/eval/qa_dataset.json under the project root.

    Returns:
        List of QAItem dicts, each with id, source_doc, question, ground_truth.
    """
    dataset_path = path or DEFAULT_DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Q&A dataset not found: {dataset_path}")

    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)

    items = data["items"]
    required_keys = {"id", "source_doc", "question", "ground_truth"}
    for item in items:
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Q&A item {item.get('id', '?')} missing keys: {missing}")

    return items
