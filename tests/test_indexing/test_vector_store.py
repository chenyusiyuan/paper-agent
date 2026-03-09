from pathlib import Path

import numpy as np

from app.indexing.vector_store import VectorStore


def test_vector_store_add_search_and_load(monkeypatch: object, tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "vector.faiss"), "dummy-model")

    def fake_encode(texts: list[str]) -> np.ndarray:
        mapping = {
            "cat research": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "dog analysis": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "orange experiment": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "cat query": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        return np.vstack([mapping[text] for text in texts])

    monkeypatch.setattr(store, "_encode", fake_encode)

    store.add(
        ["cat research", "dog analysis", "orange experiment"],
        [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}],
    )
    results = store.search("cat query", top_k=2)

    assert results[0]["text"] == "cat research"
    assert results[0]["metadata"] == {"doc_id": "d1"}

    store.save()
    loaded = VectorStore(str(tmp_path / "vector.faiss"), "dummy-model")
    monkeypatch.setattr(loaded, "_encode", fake_encode)
    loaded_results = loaded.search("cat query", top_k=1)

    assert loaded_results[0]["metadata"] == {"doc_id": "d1"}


def test_vector_store_empty_search(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "empty.faiss"), "dummy-model")
    assert store.search("anything") == []
