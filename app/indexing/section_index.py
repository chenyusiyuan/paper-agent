from __future__ import annotations

import pickle
from pathlib import Path

from app.core.schemas import SectionChunk
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_store import VectorStore


class SectionIndex:
    def __init__(
        self,
        data_dir: str,
        embedding_model: str,
        embedding_batch_size: int = 4,
        embedding_max_seq_length: int | None = 512,
    ) -> None:
        indexes_dir = Path(data_dir) / "indexes"
        self.vector_store = VectorStore(
            str(indexes_dir / "section_dense.faiss"),
            embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_max_seq_length=embedding_max_seq_length,
        )
        self.bm25_index = BM25Index(str(indexes_dir / "section_bm25.pkl"))
        self.metadata_path = indexes_dir / "section_meta.pkl"
        self.chunk_store: dict[str, SectionChunk] = {}

        if self.metadata_path.exists():
            self.load()

    def build(self, chunks: list[SectionChunk]) -> None:
        self.vector_store.reset()
        self.bm25_index.reset()
        self.chunk_store = {}

        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        doc_ids: list[str] = []
        for chunk in chunks:
            retrieval_text = f"{chunk.section_title}: {chunk.text}".strip()
            texts.append(retrieval_text)
            metadatas.append({"chunk_id": chunk.chunk_id, "paper_id": chunk.paper_id})
            doc_ids.append(chunk.chunk_id)
            self.chunk_store[chunk.chunk_id] = chunk

        self.vector_store.add(texts, metadatas)
        self.bm25_index.add(texts, doc_ids)
        self.save()

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        results = self.vector_store.search(query, top_k=top_k)
        ranked: list[tuple[str, float]] = []
        for item in results:
            metadata = item["metadata"]
            if isinstance(metadata, dict):
                chunk_id = metadata.get("chunk_id")
                if isinstance(chunk_id, str):
                    ranked.append((chunk_id, float(item["score"])))
        return ranked

    def search_sparse(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        results = self.bm25_index.search(query, top_k=top_k)
        return [
            (str(item["doc_id"]), float(item["score"]))
            for item in results
        ]

    def get_by_id(self, chunk_id: str) -> SectionChunk | None:
        return self.chunk_store.get(chunk_id)

    def get_by_paper(self, paper_id: str) -> list[SectionChunk]:
        return sorted(
            [chunk for chunk in self.chunk_store.values() if chunk.paper_id == paper_id],
            key=lambda chunk: chunk.order_in_paper,
        )

    def filter_chunk_ids(
        self,
        paper_id: str | None = None,
        target_sections: list[str] | None = None,
    ) -> set[str] | None:
        if paper_id is None and target_sections is None:
            return None
        target_set = set(target_sections or [])
        chunk_ids: set[str] = set()
        for chunk_id, chunk in self.chunk_store.items():
            if paper_id is not None and chunk.paper_id != paper_id:
                continue
            if target_set and chunk.section_type not in target_set:
                continue
            chunk_ids.add(chunk_id)
        return chunk_ids

    def save(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save()
        self.bm25_index.save()
        with self.metadata_path.open("wb") as file_obj:
            pickle.dump(self.chunk_store, file_obj)

    def load(self) -> None:
        self.vector_store.load()
        self.bm25_index.load()
        if self.metadata_path.exists():
            with self.metadata_path.open("rb") as file_obj:
                self.chunk_store = pickle.load(file_obj)
