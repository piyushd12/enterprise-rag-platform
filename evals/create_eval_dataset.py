import asyncio
import json
import argparse
import httpx

API_BASE = "http://localhost:8000"

DEV_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiZGFlZWI0Ni1hMmM5LTQ3YjEtYjZiOS00ZGQwYWM3YTRlMGQiLCJleHAiOjE3ODE4NDY0OTQsImlhdCI6MTc4MTc2MDA5NH0.VCdqpAd-nEZBHEteKhtBERnIbdX3qavxGmdPiS4A2Ys"

HEADERS = {"Authorization": f"Bearer {DEV_TOKEN}"}


async def get_chunks_for_workspace(workspace_id: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(
            f"{API_BASE}/workspaces/{workspace_id}/documents",
            headers=HEADERS,
        )
        r.raise_for_status()
        documents = r.json()

        all_chunks = []
        for doc in documents:
            if doc["status"] != "chunked":
                print(f"  Skipping '{doc['title']}' (status: {doc['status']})")
                continue
            r2 = await client.get(
                f"{API_BASE}/workspaces/{workspace_id}/documents/{doc['id']}/chunks",
                headers=HEADERS,
            )
            if r2.status_code == 200:
                chunks = r2.json()
                for chunk in chunks:
                    chunk["doc_title"] = doc["title"]
                all_chunks.extend(chunks)
                print(f"  Loaded {len(chunks)} chunks from '{doc['title']}'")

        return all_chunks


async def generate_qa_pairs(
    chunks: list[dict],
    workspace_id: str,
    num_questions: int = 20,
) -> list[dict]:
    import random
    random.seed(42)
    sampled = random.sample(chunks, min(num_questions, len(chunks)))
    qa_pairs = []

    async with httpx.AsyncClient(timeout=60) as client:
        for i, chunk in enumerate(sampled):
            print(f"  Generating Q&A {i+1}/{len(sampled)}...")

            prompt = (
                f"Read this text and generate ONE specific factual question "
                f"that can be answered directly from it. "
                f"Then provide the exact answer from the text.\n\n"
                f"TEXT:\n{chunk['chunk_text']}\n\n"
                f"Respond in this exact format:\n"
                f"QUESTION: <your question here>\n"
                f"ANSWER: <the answer from the text>"
            )

            try:
                r = await client.post(
                    f"{API_BASE}/workspaces/{workspace_id}/chat",
                    json={"query": prompt, "conversation_id": None},
                    headers=HEADERS,
                )
                if r.status_code != 200:
                    continue
                
                print(f"Question generated -> {r.stream}")

                response_text = r.json()["answer"]

                lines = response_text.strip().split("\n")
                question = None
                answer = None
                for line in lines:
                    if line.startswith("QUESTION:"):
                        question = line.replace("QUESTION:", "").strip()
                    elif line.startswith("ANSWER:"):
                        answer = line.replace("ANSWER:", "").strip()

                if question and answer:
                    qa_pairs.append({
                        "question": question,
                        "ground_truth": answer,
                        "reference_chunk": chunk["chunk_text"],
                        "document_id": chunk["document_id"],
                        "doc_title": chunk["doc_title"],
                        "page_num": chunk["page_num"],
                    })

            except Exception as e:
                print(f"    ⚠ Failed to generate Q&A for chunk: {e}")
                continue

    return qa_pairs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--output", default="evals/eval_dataset.json")
    parser.add_argument("--num-questions", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading chunks from workspace {args.workspace_id}...")
    chunks = await get_chunks_for_workspace(args.workspace_id)
    print(f"Found {len(chunks)} total chunks\n")

    if not chunks:
        print("No chunks found. Upload and process documents first.")
        return

    print(f"Generating {args.num_questions} Q&A pairs...")
    qa_pairs = await generate_qa_pairs(chunks, args.workspace_id, args.num_questions)

    with open(args.output, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"\nSaved {len(qa_pairs)} Q&A pairs to {args.output}")
    print("Commit this file — it is your permanent benchmark.")


if __name__ == "__main__":
    asyncio.run(main())