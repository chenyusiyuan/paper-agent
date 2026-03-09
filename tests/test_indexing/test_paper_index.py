from pathlib import Path

import numpy as np

from app.core.schemas import PaperMetadata
from app.indexing.paper_index import PaperIndex


def make_paper(paper_id: str, title: str, abstract: str, keywords: list[str]) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Author One"],
        year=2024,
        venue="ACL",
        abstract=abstract,
        keywords=keywords,
        section_titles=["Introduction", "Method", "Conclusion"],
    )


def test_paper_index_build_search_and_get(monkeypatch: object, tmp_path: Path) -> None:
    paper_index = PaperIndex(str(tmp_path), "dummy-model")

    def fake_encode(texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    1.0 if "transformer" in lowered or "attention" in lowered else 0.0,
                    1.0 if "graph" in lowered or "molecule" in lowered else 0.0,
                    1.0 if "vision" in lowered or "image" in lowered else 0.0,
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector) or 1.0
            vectors.append((vector / norm).astype(np.float32))
        return np.vstack(vectors)

    monkeypatch.setattr(paper_index.vector_store, "_encode", fake_encode)

    papers = [
        make_paper("p1", "Transformer Reasoning", "Attention improves reasoning", ["transformer"]),
        make_paper("p2", "Graph Molecules", "Graph learning on molecule tasks", ["graph", "molecule"]),
        make_paper("p3", "Vision Retrieval", "Image retrieval with vision transformers", ["vision"]),
        make_paper("p4", "Speech Models", "Audio understanding for speech", ["speech"]),
        make_paper("p5", "Planning Agents", "Planning for agent systems", ["agent"]),
    ]

    paper_index.build(papers)

    dense_results = paper_index.search_dense("transformer attention", top_k=2)
    sparse_results = paper_index.search_sparse("graph molecule", top_k=2)

    assert dense_results[0][0] == "p1"
    assert sparse_results[0][0] == "p2"
    assert paper_index.get_by_id("p2") is not None
    assert paper_index.get_by_id("nonexistent") is None

    loaded = PaperIndex(str(tmp_path), "dummy-model")
    assert loaded.get_by_id("p1") is not None
