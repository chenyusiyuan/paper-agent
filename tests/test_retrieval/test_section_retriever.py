from app.core.schemas import SectionChunk
from app.retrieval.section_retriever import SectionRetriever


def make_chunk(
    paper_id: str,
    section_type: str,
    order: int,
    text: str,
    granularity: str = "fine",
    level: int = 1,
    parent_chunk_id: str | None = None,
) -> SectionChunk:
    return SectionChunk(
        chunk_id=f"{paper_id}_{section_type}_{order}",
        paper_id=paper_id,
        section_type=section_type,
        section_title=section_type.title(),
        section_path=section_type.title(),
        text=text,
        page_start=0,
        page_end=0,
        order_in_paper=order,
        level=level,
        parent_chunk_id=parent_chunk_id,
        granularity=granularity,
    )


class StubSectionIndex:
    def __init__(self) -> None:
        self.chunks = [
            make_chunk("paper-1", "abstract", 0, "summary"),
            make_chunk("paper-1", "introduction", 1, "problem setup"),
            make_chunk("paper-1", "method", 2, "training method details"),
            make_chunk("paper-1", "method", 6, "optimizer and scheduler"),
            make_chunk("paper-1", "experiment", 3, "benchmark result"),
            make_chunk("paper-1", "experiment", 7, "ablation study"),
            make_chunk("paper-1", "conclusion", 4, "takeaways"),
            make_chunk("paper-1", "other", 5, "appendix note"),
            make_chunk("paper-2", "method", 0, "other paper method"),
        ]
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

    def get_by_paper(self, paper_id: str) -> list[SectionChunk]:
        return [chunk for chunk in self.chunks if chunk.paper_id == paper_id]

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return [
            ("paper-1_method_2", 0.9),
            ("paper-1_experiment_3", 0.8),
            ("paper-2_method_0", 0.7),
            ("paper-1_introduction_1", 0.6),
            ("paper-1_method_6", 0.5),
            ("paper-1_experiment_7", 0.4),
        ]

    def search_sparse(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return [
            ("paper-1_experiment_3", 8.0),
            ("paper-1_method_2", 7.5),
            ("paper-1_method_6", 7.0),
            ("paper-1_experiment_7", 6.5),
            ("paper-1_conclusion_4", 6.0),
        ]

    def get_by_id(self, chunk_id: str) -> SectionChunk | None:
        return self.chunk_by_id.get(chunk_id)


class StubReranker:
    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        assert query == "training details"
        # reranker favors method_2 (idx 0) then experiment_3 (idx 1)
        return [(0, 0.9), (1, 0.8)][:top_k]


def test_section_retriever_filters_by_paper_and_section() -> None:
    """Full pipeline: filter → search → RRF → rerank. Candidate count (4) > top_k (2)."""
    retriever = SectionRetriever(StubSectionIndex(), StubReranker())  # type: ignore[arg-type]

    results = retriever.retrieve(
        "training details",
        paper_id="paper-1",
        target_sections=["method", "experiment"],
        top_k=2,
    )

    # Reranker returns idx 0 (method_2) then idx 1 (experiment_3) from the fused list.
    assert len(results) == 2
    assert results[0].section_type == "method"
    assert results[1].section_type == "experiment"


def test_section_retriever_short_circuit_when_candidates_small() -> None:
    """When candidate count (2) <= top_k (3), return immediately without search/rerank."""
    retriever = SectionRetriever(StubSectionIndex(), StubReranker())  # type: ignore[arg-type]

    results = retriever.retrieve(
        "method",
        paper_id="paper-1",
        target_sections=["method"],
        top_k=3,
    )

    assert len(results) == 2
    assert all(chunk.section_type == "method" for chunk in results)
