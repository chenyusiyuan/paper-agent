from __future__ import annotations

import pickle
from pathlib import Path

from app.core.schemas import PaperMetadata
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_store import VectorStore


class PaperIndex:
    def __init__(
        self,
        data_dir: str,
        embedding_model: str,
        embedding_batch_size: int = 4,
        embedding_max_seq_length: int | None = 512,
    ) -> None:
        indexes_dir = Path(data_dir) / "indexes"
        self.vector_store = VectorStore(
            str(indexes_dir / "paper_dense.faiss"),
            embedding_model,
            embedding_batch_size=embedding_batch_size,
            embedding_max_seq_length=embedding_max_seq_length,
        )
        self.bm25_index = BM25Index(str(indexes_dir / "paper_bm25.pkl"))
        self.metadata_path = indexes_dir / "paper_meta.pkl"
        self.metadata_store: dict[str, PaperMetadata] = {}

        if self.metadata_path.exists():
            self.load()

    def build(self, papers: list[PaperMetadata]) -> None:
        self.vector_store.reset()
        self.bm25_index.reset()
        self.metadata_store = {}

        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        doc_ids: list[str] = []
        for paper in papers:
            retrieval_text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".strip()
            texts.append(retrieval_text)
            metadatas.append({"paper_id": paper.paper_id})
            doc_ids.append(paper.paper_id)
            self.metadata_store[paper.paper_id] = paper

        self.vector_store.add(texts, metadatas)
        self.bm25_index.add(texts, doc_ids)
        self.save()

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        results = self.vector_store.search(query, top_k=top_k)
        ranked: list[tuple[str, float]] = []
        for item in results:
            metadata = item["metadata"]
            if isinstance(metadata, dict):
                paper_id = metadata.get("paper_id")
                if isinstance(paper_id, str):
                    ranked.append((paper_id, float(item["score"])))
        return ranked

    def search_sparse(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        results = self.bm25_index.search(query, top_k=top_k)
        return [
            (str(item["doc_id"]), float(item["score"]))
            for item in results
        ]

    def get_by_id(self, paper_id: str) -> PaperMetadata | None:
        return self.metadata_store.get(paper_id)

    def save(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save()
        self.bm25_index.save()
        with self.metadata_path.open("wb") as file_obj:
            pickle.dump(self.metadata_store, file_obj)

    def load(self) -> None:
        self.vector_store.load()
        self.bm25_index.load()
        if self.metadata_path.exists():
            with self.metadata_path.open("rb") as file_obj:
                self.metadata_store = pickle.load(file_obj)
