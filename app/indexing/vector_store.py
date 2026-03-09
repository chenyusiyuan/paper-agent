from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, index_path: str, embedding_model: str) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = self.index_path.with_suffix(".pkl")
        self.embedding_model = embedding_model
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._vector_dim: int | None = None
        self._entries: list[dict[str, object]] = []

        if self.index_path.exists() and self.metadata_path.exists():
            self.load()

    def reset(self) -> None:
        self._index = None
        self._vector_dim = None
        self._entries = []

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / norms).astype(np.float32)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self._vector_dim or 0
            return np.zeros((0, dimension), dtype=np.float32)

        model = self._load_model()
        encoded = model.encode(texts, convert_to_numpy=True)
        embeddings = self._normalize(np.asarray(encoded, dtype=np.float32))

        if embeddings.size > 0 and self._vector_dim is None:
            self._vector_dim = int(embeddings.shape[1])
        return embeddings

    def add(self, texts: list[str], metadatas: list[dict[str, object]]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have the same length")
        if not texts:
            return

        embeddings = self._encode(texts)
        if self._vector_dim is None and embeddings.size > 0:
            self._vector_dim = int(embeddings.shape[1])

        if self._index is None:
            if self._vector_dim is None:
                raise ValueError("vector dimension is not initialized")
            self._index = faiss.IndexFlatIP(self._vector_dim)
        self._index.add(embeddings)

        for text, metadata in zip(texts, metadatas):
            self._entries.append({"text": text, "metadata": metadata})

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        if not self._entries or self._index is None:
            return []

        query_vector = self._encode([query])
        if query_vector.size == 0:
            return []

        scores, indices = self._index.search(query_vector, min(top_k, len(self._entries)))

        results: list[dict[str, object]] = []
        for index, score in zip(indices[0].tolist(), scores[0].tolist()):
            if index < 0 or index >= len(self._entries):
                continue
            entry = self._entries[index]
            results.append(
                {
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "score": float(score),
                }
            )
        return results

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))

        payload = {
            "entries": self._entries,
            "vector_dim": self._vector_dim,
        }
        with self.metadata_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

    def load(self) -> None:
        if self.metadata_path.exists():
            with self.metadata_path.open("rb") as file_obj:
                payload = pickle.load(file_obj)
            self._entries = list(payload.get("entries", []))
            self._vector_dim = payload.get("vector_dim")

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            if self._vector_dim is None:
                self._vector_dim = int(self._index.d)
