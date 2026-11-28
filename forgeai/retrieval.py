"""BM25, embedding, and hybrid document retrieval."""

import hashlib
import logging
import re
from pathlib import Path

import faiss
import numpy as np
import ollama
from ollama import EmbedResponse
from rank_bm25 import BM25Okapi

from forgeai import config
from forgeai.loader import Chunk
from forgeai.schemas import RetrievalConfig

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", text.lower()) if len(token) > 1]


class Retriever:
    """Search document chunks using the configured retrieval method."""

    _EMBED_BATCH_SIZE = 32
    _RRF_K = 60

    def __init__(self, chunks: list[Chunk], cfg: RetrievalConfig):
        self.chunks = chunks
        self.cfg = cfg
        self._bm25 = (
            BM25Okapi([_tokenize(chunk.text) for chunk in chunks])
            if chunks
            else None
        )

    def _bm25_ranking(self, query: str) -> list[tuple[int, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        return sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda result: (-result[1], result[0]),
        )

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _response_embeddings(
        response: EmbedResponse | dict[str, list[list[float]]],
    ) -> list[list[float]]:
        if isinstance(response, dict):
            embeddings = response["embeddings"]
        else:
            embeddings = response.embeddings
        return embeddings

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        cache_dir = Path(".cache") / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        vectors: dict[str, np.ndarray] = {}
        missing: dict[str, str] = {}
        for text in texts:
            key = self._cache_key(text)
            if key in vectors or key in missing:
                continue
            cache_path = cache_dir / f"{key}.npy"
            if cache_path.exists():
                try:
                    vectors[key] = np.asarray(
                        np.load(cache_path, allow_pickle=False), dtype=np.float32
                    )
                    continue
                except (OSError, ValueError) as error:
                    logger.warning("Could not read embedding cache %s: %s", cache_path, error)
            missing[key] = text

        missing_items = list(missing.items())
        for start in range(0, len(missing_items), self._EMBED_BATCH_SIZE):
            batch = missing_items[start : start + self._EMBED_BATCH_SIZE]
            response = ollama.embed(
                model=config.EMBED_MODEL,
                input=[text for _, text in batch],
            )
            batch_vectors = self._response_embeddings(response)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    "Ollama returned a different number of embeddings than requested"
                )
            for (key, _), vector in zip(batch, batch_vectors):
                embedding = np.asarray(vector, dtype=np.float32)
                if embedding.ndim != 1:
                    raise ValueError("Ollama returned an embedding that is not one-dimensional")
                vectors[key] = embedding
                try:
                    np.save(cache_dir / f"{key}.npy", embedding)
                except OSError as error:
                    logger.warning("Could not write embedding cache for %s: %s", key, error)

        try:
            return np.stack([vectors[self._cache_key(text)] for text in texts]).astype(
                np.float32
            )
        except ValueError as error:
            raise ValueError("Cached or returned embeddings have inconsistent dimensions") from error

    def _embedding_ranking(self, query: str) -> list[tuple[int, float]]:
        if not self.chunks:
            return []
        document_vectors = self._get_embeddings([chunk.text for chunk in self.chunks])
        query_vector = self._get_embeddings([query])
        if query_vector.shape[1] != document_vectors.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match")

        faiss.normalize_L2(document_vectors)
        faiss.normalize_L2(query_vector)
        index = faiss.IndexFlatIP(document_vectors.shape[1])
        index.add(document_vectors)
        scores, indices = index.search(query_vector, len(self.chunks))
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks and their scores for a query."""
        if not self.chunks or self.cfg.top_k <= 0:
            return []

        if self.cfg.method == "bm25":
            ranking = self._bm25_ranking(query)
        else:
            try:
                embedding_ranking = self._embedding_ranking(query)
            except Exception as error:
                logger.warning(
                    "Embedding retrieval failed; falling back to BM25: %s", error
                )
                ranking = self._bm25_ranking(query)
            else:
                if self.cfg.method == "embedding":
                    ranking = embedding_ranking
                else:
                    bm25_ranking = self._bm25_ranking(query)
                    fused_scores: dict[int, float] = {}
                    for ranked_items in (bm25_ranking, embedding_ranking):
                        for rank, (chunk_index, _) in enumerate(ranked_items, start=1):
                            fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + (
                                1.0 / (self._RRF_K + rank)
                            )
                    ranking = sorted(
                        fused_scores.items(),
                        key=lambda result: (-result[1], result[0]),
                    )

        return [
            (self.chunks[index], score)
            for index, score in ranking[: self.cfg.top_k]
        ]

    def retrieved_doc_ids(self, query: str) -> list[str]:
        """Return unique source document IDs in retrieval rank order."""
        doc_ids: list[str] = []
        seen: set[str] = set()
        for retrieved_chunk, _ in self.search(query):
            if retrieved_chunk.doc_id not in seen:
                seen.add(retrieved_chunk.doc_id)
                doc_ids.append(retrieved_chunk.doc_id)
        return doc_ids
