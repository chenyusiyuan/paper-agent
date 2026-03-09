from pathlib import Path

from app.indexing.bm25_index import BM25Index


def test_bm25_index_search_and_load(tmp_path: Path) -> None:
    index = BM25Index(str(tmp_path / "bm25.pkl"))
    index.add(
        [
            "transformer attention language model",
            "graph neural network for molecules",
            "vision transformer image classification",
        ],
        ["p1", "p2", "p3"],
    )

    results = index.search("graph molecules", top_k=2)

    assert results[0]["doc_id"] == "p2"
    assert results[0]["text"] == "graph neural network for molecules"

    index.save()
    loaded = BM25Index(str(tmp_path / "bm25.pkl"))
    loaded_results = loaded.search("image classification", top_k=1)

    assert loaded_results[0]["doc_id"] == "p3"
