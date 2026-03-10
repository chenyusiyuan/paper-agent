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


def test_vector_store_reuses_loaded_model(monkeypatch: object, tmp_path: Path) -> None:
    init_calls: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            init_calls.append(model_name)
            self.max_seq_length: int | None = None

        def encode(
            self,
            texts: list[str],
            batch_size: int = 32,
            convert_to_numpy: bool = True,
        ) -> np.ndarray:
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr("app.indexing.vector_store.SentenceTransformer", FakeSentenceTransformer)
    VectorStore._shared_models = {}

    store_one = VectorStore(str(tmp_path / "one.faiss"), "shared-model")
    store_two = VectorStore(str(tmp_path / "two.faiss"), "shared-model")

    store_one.add(["paper text"], [{"doc_id": "p1"}])
    store_two.add(["section text"], [{"doc_id": "s1"}])

    assert init_calls == ["shared-model"]


def test_vector_store_uses_configured_batch_and_max_seq_length(monkeypatch: object, tmp_path: Path) -> None:
    encode_calls: list[dict[str, object]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.max_seq_length: int | None = None

        def encode(
            self,
            texts: list[str],
            batch_size: int = 32,
            convert_to_numpy: bool = True,
        ) -> np.ndarray:
            encode_calls.append(
                {
                    "texts": texts,
                    "batch_size": batch_size,
                    "convert_to_numpy": convert_to_numpy,
                    "max_seq_length": self.max_seq_length,
                }
            )
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr("app.indexing.vector_store.SentenceTransformer", FakeSentenceTransformer)
    VectorStore._shared_models = {}

    store = VectorStore(
        str(tmp_path / "configured.faiss"),
        "configured-model",
        embedding_batch_size=2,
        embedding_max_seq_length=256,
    )

    store.add(["one", "two", "three"], [{"doc_id": "1"}, {"doc_id": "2"}, {"doc_id": "3"}])

    assert encode_calls == [
        {
            "texts": ["one", "two", "three"],
            "batch_size": 2,
            "convert_to_numpy": True,
            "max_seq_length": 256,
        }
    ]
