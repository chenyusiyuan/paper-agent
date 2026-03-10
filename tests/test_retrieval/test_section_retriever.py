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
        return [(0, 0.9), (1, 0.8)][:top_k]


def test_section_retriever_filters_by_paper_and_section() -> None:
    """Full pipeline without target_sections still uses search → RRF → rerank."""
    retriever = SectionRetriever(StubSectionIndex(), StubReranker())  # type: ignore[arg-type]

    results = retriever.retrieve(
        "training details",
        paper_id="paper-1",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].section_type == "method"
    assert results[1].section_type == "experiment"


def test_section_retriever_target_sections_returns_in_document_order() -> None:
    """When target_sections is set, return matching chunks in document order."""
    retriever = SectionRetriever(StubSectionIndex(), StubReranker())  # type: ignore[arg-type]

    results = retriever.retrieve(
        "method",
        paper_id="paper-1",
        target_sections=["method"],
        top_k=3,
    )

    assert len(results) == 2
    assert all(chunk.section_type == "method" for chunk in results)
    assert [chunk.order_in_paper for chunk in results] == [2, 6]


class SmallSectionIndex:
    def __init__(self) -> None:
        self.chunks = [
            make_chunk("paper-1", "abstract", 0, "summary"),
            make_chunk("paper-1", "method", 1, "training details"),
        ]
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

    def get_by_paper(self, paper_id: str) -> list[SectionChunk]:
        return [chunk for chunk in self.chunks if chunk.paper_id == paper_id]

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return [("paper-1_abstract_0", 0.9), ("paper-1_method_1", 0.8)]

    def search_sparse(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return []

    def get_by_id(self, chunk_id: str) -> SectionChunk | None:
        return self.chunk_by_id.get(chunk_id)


class SmallStubReranker:
    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        return [(1, 0.9), (0, 0.8)][:top_k]


def test_section_retriever_reranks_when_no_target_sections_and_few_candidates() -> None:
    retriever = SectionRetriever(SmallSectionIndex(), SmallStubReranker())  # type: ignore[arg-type]

    results = retriever.retrieve(
        "training details",
        paper_id="paper-1",
        top_k=5,
    )

    assert len(results) == 2
    assert results[0].section_type == "method"
    assert results[1].section_type == "abstract"
