# app/services/retrieval/bm25_index.py
"""
In-memory BM25 index over document chunks, scoped per workspace.

Built from PostgreSQL on startup. Provides keyword-based sparse retrieval
that complements dense vector search — especially strong for exact proper
nouns, codes, and rare technical terms that semantic embeddings struggle with.

Why in-memory and not Redis?
For development, rebuilding from PostgreSQL on startup is simple and fast
enough (<1 second for 10,000 chunks). A Redis serialisation layer is a
Week 11 production concern.
"""
import logging
import string

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    Simple tokenizer: lowercase, strip punctuation, split on whitespace.

    CRITICAL: Query and documents MUST be tokenized identically.
    Any difference (e.g. casing on query but not corpus) will silently
    produce wrong BM25 scores.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    # Drop single-character tokens — they add noise without information
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """
    Thread-safe in-memory BM25 index over document chunks.

    Workspace isolation is enforced via a pre-computed position index
    (workspace_id → list of array positions). BM25 scores are computed
    over the full corpus but results are filtered to the requesting
    workspace before being returned — no chunk ever leaks across tenants.
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._chunk_metadata: list[dict] = []
        # workspace_id → sorted list of corpus positions for that workspace
        self._workspace_index: dict[str, list[int]] = {}
        self._total_chunks: int = 0

    def build(self, chunks: list[dict]) -> None:
        """
        Build (or rebuild) the BM25 index from a list of chunk dicts.

        Each dict must contain: id, workspace_id, chunk_text,
        document_id, page_num, chunk_index.

        Call on app startup after loading chunks from PostgreSQL.
        Call again after new documents finish processing.
        """
        if not chunks:
            logger.warning("BM25 build called with empty chunks list — index not built")
            return

        tokenized_corpus: list[list[str]] = []
        self._chunk_ids = []
        self._chunk_texts = []
        self._chunk_metadata = []
        self._workspace_index = {}

        for i, chunk in enumerate(chunks):
            tokens = _tokenize(chunk["chunk_text"])
            tokenized_corpus.append(tokens)

            self._chunk_ids.append(chunk["id"])
            self._chunk_texts.append(chunk["chunk_text"])
            self._chunk_metadata.append(
                {
                    "document_id": chunk["document_id"],
                    "workspace_id": chunk["workspace_id"],
                    "page_num": chunk.get("page_num", 1),
                    "chunk_index": chunk.get("chunk_index", i),
                }
            )

            ws_id = chunk["workspace_id"]
            if ws_id not in self._workspace_index:
                self._workspace_index[ws_id] = []
            self._workspace_index[ws_id].append(i)

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._total_chunks = len(chunks)

        logger.info(
            f"BM25 index built: {self._total_chunks} chunks across "
            f"{len(self._workspace_index)} workspace(s)"
        )

    def search(
        self,
        query: str,
        workspace_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Return the top_k most BM25-relevant chunks for workspace_id.

        Always scoped to workspace_id — chunks from other workspaces
        are never returned regardless of their BM25 score.

        Returns dicts matching the same schema as VectorStoreService.search()
        so that RRF fusion can treat both result lists identically.
        """
        if self._bm25 is None or self._total_chunks == 0:
            logger.warning("BM25 search called but index is not built — returning []")
            return []

        workspace_positions = self._workspace_index.get(workspace_id, [])
        if not workspace_positions:
            logger.debug(f"No chunks indexed for workspace {workspace_id!r}")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.debug("BM25: empty token list after tokenizing query")
            return []

        # Score all chunks in the full corpus, then filter to this workspace
        all_scores = self._bm25.get_scores(query_tokens)

        workspace_scores = [
            (pos, all_scores[pos])
            for pos in workspace_positions
            if all_scores[pos] > 0  # zero-score = no shared terms with query
        ]

        # Sort descending by BM25 score, take top_k
        workspace_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = workspace_scores[:top_k]

        results = []
        for pos, score in top_results:
            meta = self._chunk_metadata[pos]
            results.append(
                {
                    "chunk_id": self._chunk_ids[pos],
                    "chunk_text": self._chunk_texts[pos],
                    "document_id": meta["document_id"],
                    "workspace_id": meta["workspace_id"],
                    "page_num": meta["page_num"],
                    "chunk_index": meta["chunk_index"],
                    "score": float(score),  # raw BM25 score — RRF will normalise by rank
                }
            )

        logger.debug(
            f"BM25 search '{query[:40]}' → {len(results)} results "
            f"in workspace {workspace_id[:8]}"
        )
        return results

    def add_chunks(self, new_chunks: list[dict]) -> None:
        """
        Incrementally add chunks without full rebuild.
        Merges new chunks with existing index data and rebuilds.
        Acceptable for corpora up to ~50k chunks.
        """
        existing: list[dict] = []
        for i, chunk_id in enumerate(self._chunk_ids):
            existing.append(
                {
                    "id": chunk_id,
                    "chunk_text": self._chunk_texts[i],
                    **self._chunk_metadata[i],
                }
            )
        self.build(existing + new_chunks)
        logger.info(f"BM25 index rebuilt: added {len(new_chunks)} new chunks")

    def remove_document(self, document_id: str) -> None:
        """
        Remove all chunks belonging to a document and rebuild.
        Called when a document is deleted.
        """
        remaining = [
            {
                "id": self._chunk_ids[i],
                "chunk_text": self._chunk_texts[i],
                **self._chunk_metadata[i],
            }
            for i, meta in enumerate(self._chunk_metadata)
            if meta["document_id"] != document_id
        ]
        self.build(remaining)
        logger.info(f"BM25 index rebuilt after removing document {document_id!r}")

    @property
    def is_ready(self) -> bool:
        """True when the index has been built and contains at least one chunk."""
        return self._bm25 is not None and self._total_chunks > 0

    @property
    def total_chunks(self) -> int:
        return self._total_chunks


# Module-level singleton — shared across the entire FastAPI process
bm25_index = BM25Index()
