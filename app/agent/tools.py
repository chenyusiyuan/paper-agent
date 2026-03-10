from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.schemas import PaperMetadata, SectionChunk

if TYPE_CHECKING:
    from app.indexing.paper_index import PaperIndex
    from app.retrieval.paper_retriever import PaperRetriever
    from app.retrieval.section_retriever import SectionRetriever


class AgentTools:
    def __init__(
        self,
        paper_retriever: PaperRetriever,
        section_retriever: SectionRetriever,
        paper_index: PaperIndex,
    ) -> None:
        self.paper_retriever = paper_retriever
        self.section_retriever = section_retriever
        self.paper_index = paper_index

    def search_papers(self, query: str, top_k: int = 3) -> list[PaperMetadata]:
        return self.paper_retriever.retrieve(query, top_k=top_k)

    def get_paper_metadata(self, paper_id: str) -> PaperMetadata | None:
        return self.paper_index.get_by_id(paper_id)

    def retrieve_sections(
        self,
        paper_id: str,
        query: str,
        target_sections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[SectionChunk]:
        return self.section_retriever.retrieve(
            query=query,
            paper_id=paper_id,
            target_sections=target_sections,
            top_k=top_k,
        )
