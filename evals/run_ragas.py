"""
Runs RAGAS evaluation against the live RAG system.
Run BEFORE and AFTER changes to measure improvement.

Usage:
    uv run python evals/run_ragas.py \
        --workspace-id YOUR_ID \
        --dataset evals/eval_dataset.json \
        --output evals/results_dense_only.json \
        --label "Dense Only"
"""
import asyncio
import json
import argparse
import httpx
from datetime import datetime

API_BASE = "http://localhost:8000"
DEV_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiZGFlZWI0Ni1hMmM5LTQ3YjEtYjZiOS00ZGQwYWM3YTRlMGQiLCJleHAiOjE3ODE4NDY0OTQsImlhdCI6MTc4MTc2MDA5NH0.VCdqpAd-nEZBHEteKhtBERnIbdX3qavxGmdPiS4A2Ys"
HEADERS = {"Authorization": f"Bearer {DEV_TOKEN}"}


async def run_rag_for_question(
    client: httpx.AsyncClient,
    workspace_id: str,
    question: str,
) -> dict:
    r = await client.post(
        f"{API_BASE}/workspaces/{workspace_id}/chat",
        json={"query": question, "conversation_id": None},
        headers=HEADERS,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "answer": data["answer"],
        "contexts": [s["chunk_text"] for s in data.get("sources", [])],
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--dataset", default="evals/eval_dataset.json")
    parser.add_argument("--output", default="evals/results.json")
    parser.add_argument("--label", default="Evaluation Run")
    args = parser.parse_args()

    with open(args.dataset) as f:
        qa_pairs = json.load(f)

    print(f"Running RAGAS eval: '{args.label}'")
    print(f"Questions: {len(qa_pairs)}\n")

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i, qa in enumerate(qa_pairs):
            print(f"  [{i+1}/{len(qa_pairs)}] {qa['question'][:60]}...")
            try:
                result = await run_rag_for_question(
                    client, args.workspace_id, qa["question"]
                )
                questions.append(qa["question"])
                ground_truths.append(qa["ground_truth"])
                answers.append(result["answer"])
                contexts.append(result["contexts"])
            except Exception as e:
                print(f"    ⚠ Failed: {e}")

    print("\nCalculating RAGAS metrics...")
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        context_recall,
        context_precision,
        faithfulness,
        answer_relevancy,
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    import os
    from dotenv import load_dotenv
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

    result = evaluate(
        dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=None,
    )

    scores = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "context_recall": round(float(result["context_recall"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
    }

    print(f"\n{'='*50}")
    print(f"Results: {args.label}")
    print(f"{'='*50}")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
    print(f"{'='*50}\n")

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"✅ Saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())