from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.schemas import SectionChunk
from app.indexing.section_index import SectionIndex
from app.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from app.retrieval.reranker import Reranker


class SectionRetriever:
    def __init__(self, section_index: SectionIndex, reranker: Reranker) -> None:
        self.section_index = section_index
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        paper_id: str,
        target_sections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[SectionChunk]:
        candidates = self.section_index.get_by_paper(paper_id)
        if target_sections:
            allowed_types = set(target_sections)
            candidates = [
                chunk for chunk in candidates if chunk.section_type in allowed_types
            ]

        if not candidates:
            return []
        if target_sections:
            return candidates[:top_k]
        if len(candidates) == 1:
            return candidates
        if len(candidates) <= top_k:
            passages = [chunk.text for chunk in candidates]
            reranked = self.reranker.rerank(query, passages, top_k=len(candidates))
            return [
                candidates[index]
                for index, _ in reranked
                if 0 <= index < len(candidates)
            ]

        candidate_ids = {chunk.chunk_id for chunk in candidates}
        dense = [
            (chunk_id, score)
            for chunk_id, score in self.section_index.search_dense(query, top_k=max(top_k * 4, 20))
            if chunk_id in candidate_ids
        ]
        sparse = [
            (chunk_id, score)
            for chunk_id, score in self.section_index.search_sparse(query, top_k=max(top_k * 4, 20))
            if chunk_id in candidate_ids
        ]
        fused = reciprocal_rank_fusion([dense, sparse])

        fused_chunks: list[SectionChunk] = []
        for chunk_id, _ in fused:
            chunk = self.section_index.get_by_id(chunk_id)
            if chunk is not None:
                fused_chunks.append(chunk)

        if not fused_chunks:
            return candidates[:top_k]

        passages = [chunk.text for chunk in fused_chunks]
        reranked = self.reranker.rerank(query, passages, top_k=top_k)
        ranked_chunks = [
            fused_chunks[index]
            for index, _ in reranked
            if 0 <= index < len(fused_chunks)
        ]

        if len(ranked_chunks) >= top_k:
            return ranked_chunks[:top_k]

        seen_chunk_ids = {chunk.chunk_id for chunk in ranked_chunks}
        for chunk in candidates:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            ranked_chunks.append(chunk)
            if len(ranked_chunks) >= top_k:
                break
        return ranked_chunks
