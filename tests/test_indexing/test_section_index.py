from pathlib import Path

import numpy as np

from app.core.schemas import SectionChunk
from app.indexing.section_index import SectionIndex


def make_chunk(paper_id: str, section_type: str, order: int, text: str) -> SectionChunk:
    title = section_type.replace("_", " ").title()
    return SectionChunk(
        chunk_id=f"{paper_id}_{section_type}_{order}",
        paper_id=paper_id,
        section_type=section_type,
        section_title=title,
        section_path=title,
        text=text,
        page_start=0,
        page_end=0,
        order_in_paper=order,
        level=1,
        parent_chunk_id=None,
        granularity="fine",
    )


def test_section_index_build_and_filter(monkeypatch: object, tmp_path: Path) -> None:
    section_index = SectionIndex(str(tmp_path), "dummy-model")

    def fake_encode(texts: list[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    1.0 if "introduction" in lowered else 0.0,
                    1.0 if "training" in lowered or "method" in lowered else 0.0,
                    1.0 if "result" in lowered or "experiment" in lowered else 0.0,
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector) or 1.0
            vectors.append((vector / norm).astype(np.float32))
        return np.vstack(vectors)

    monkeypatch.setattr(section_index.vector_store, "_encode", fake_encode)

    paper_one_chunks = [
        make_chunk("paper-1", "abstract", 0, "Summary of the paper"),
        make_chunk("paper-1", "introduction", 1, "Introduction to the problem"),
        make_chunk("paper-1", "method", 2, "Training method details"),
        make_chunk("paper-1", "experiment", 3, "Experiment result analysis"),
        make_chunk("paper-1", "conclusion", 4, "Final conclusion"),
    ]
    paper_two_chunks = [
        make_chunk("paper-2", "abstract", 0, "Other summary"),
        make_chunk("paper-2", "introduction", 1, "Another introduction"),
        make_chunk("paper-2", "method", 2, "Optimization method"),
        make_chunk("paper-2", "experiment", 3, "Benchmark results"),
        make_chunk("paper-2", "conclusion", 4, "Other conclusion"),
    ]

    section_index.build(paper_one_chunks + paper_two_chunks)

    # Test get_by_paper
    paper_one = section_index.get_by_paper("paper-1")
    assert len(paper_one) == 5
    assert paper_one[0].chunk_id == "paper-1_abstract_0"

    # Test filter_chunk_ids
    filtered = section_index.filter_chunk_ids(paper_id="paper-1", target_sections=["method"])
    assert filtered is not None
    assert len(filtered) == 1
    chunk_id = next(iter(filtered))
    chunk = section_index.get_by_id(chunk_id)
    assert chunk is not None
    assert chunk.paper_id == "paper-1"
    assert chunk.section_type == "method"

    # Test search_dense returns results
    dense = section_index.search_dense("training method", top_k=3)
    assert len(dense) > 0

    # Test persistence
    loaded = SectionIndex(str(tmp_path), "dummy-model")
    assert len(loaded.get_by_paper("paper-2")) == 5
