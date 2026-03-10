from app.core.schemas import PaperMetadata
from app.retrieval.paper_retriever import PaperRetriever


class StubPaperIndex:
    def __init__(self) -> None:
        self.papers = {
            "p1": PaperMetadata(
                paper_id="p1",
                title="Transformer Agents",
                authors=["A"],
                year=2024,
                venue="ACL",
                abstract="Paper one abstract",
                keywords=["transformer"],
                section_titles=[],
            ),
            "p2": PaperMetadata(
                paper_id="p2",
                title="Graph Reasoning",
                authors=["B"],
                year=2023,
                venue="NeurIPS",
                abstract="Paper two abstract",
                keywords=["graph"],
                section_titles=[],
            ),
            "p3": PaperMetadata(
                paper_id="p3",
                title="Vision Models",
                authors=["C"],
                year=2022,
                venue="ICCV",
                abstract="Paper three abstract",
                keywords=["vision"],
                section_titles=[],
            ),
        }

    def search_dense(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return [("p1", 0.9), ("p2", 0.8), ("p3", 0.7)]

    def search_sparse(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return [("p2", 9.0), ("p1", 8.0)]

    def get_by_id(self, paper_id: str) -> PaperMetadata | None:
        return self.papers.get(paper_id)


class StubReranker:
    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        assert query == "transformer agent"
        assert passages == ["Paper one abstract", "Paper two abstract", "Paper three abstract"]
        return [(1, 0.9), (0, 0.8)][:top_k]


def test_paper_retriever_runs_full_pipeline() -> None:
    retriever = PaperRetriever(StubPaperIndex(), StubReranker())  # type: ignore[arg-type]

    results = retriever.retrieve("transformer agent", top_k=2)

    assert [paper.paper_id for paper in results] == ["p2", "p1"]
