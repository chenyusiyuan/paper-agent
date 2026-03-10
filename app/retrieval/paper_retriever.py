from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.schemas import PaperMetadata
from app.indexing.paper_index import PaperIndex
from app.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from app.retrieval.reranker import Reranker


class PaperRetriever:
    def __init__(self, paper_index: PaperIndex, reranker: Reranker) -> None:
        self.paper_index = paper_index
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 3) -> list[PaperMetadata]:
        dense = self.paper_index.search_dense(query, top_k=20)
        sparse = self.paper_index.search_sparse(query, top_k=20)
        fused = reciprocal_rank_fusion([dense, sparse])[:10]

        candidates: list[PaperMetadata] = []
        passages: list[str] = []
        for paper_id, _ in fused:
            metadata = self.paper_index.get_by_id(paper_id)
            if metadata is None:
                continue
            candidates.append(metadata)
            passages.append(metadata.abstract)

        if not candidates:
            return []

        reranked = self.reranker.rerank(query, passages, top_k=top_k)
        return [
            candidates[index]
            for index, _ in reranked
            if 0 <= index < len(candidates)
        ]
