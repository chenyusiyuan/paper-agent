from __future__ import annotations

import math
import pickle
from pathlib import Path


class BM25Index:
    def __init__(self, index_path: str) -> None:
        self.index_path = Path(index_path)
        self.texts: list[str] = []
        self.doc_ids: list[str] = []
        self.tokenized_texts: list[list[str]] = []
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75

        if self.index_path.exists():
            self.load()

    def reset(self) -> None:
        self.texts = []
        self.doc_ids = []
        self.tokenized_texts = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0.0

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in text.lower().split() if token]

    def _rebuild(self) -> None:
        self.doc_freqs = []
        document_frequency: dict[str, int] = {}

        for tokens in self.tokenized_texts:
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        corpus_size = len(self.tokenized_texts)
        total_length = sum(len(tokens) for tokens in self.tokenized_texts)
        self.avgdl = total_length / corpus_size if corpus_size else 0.0
        self.idf = {}
        for token, frequency in document_frequency.items():
            self.idf[token] = math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))

    def add(self, texts: list[str], doc_ids: list[str]) -> None:
        if len(texts) != len(doc_ids):
            raise ValueError("texts and doc_ids must have the same length")
        self.texts.extend(texts)
        self.doc_ids.extend(doc_ids)
        self.tokenized_texts.extend(self._tokenize(text) for text in texts)
        self._rebuild()

    def _score_document(self, query_tokens: list[str], index: int) -> float:
        if index >= len(self.doc_freqs):
            return 0.0
        doc_freq = self.doc_freqs[index]
        doc_length = len(self.tokenized_texts[index]) or 1
        score = 0.0
        for token in query_tokens:
            frequency = doc_freq.get(token, 0)
            if frequency == 0:
                continue
            idf = self.idf.get(token, 0.0)
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self.avgdl or 1.0)
            )
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> list[dict[str, object]]:
        query_tokens = self._tokenize(query)
        scores = [
            (index, self._score_document(query_tokens, index))
            for index in range(len(self.doc_ids))
        ]
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                "doc_id": self.doc_ids[index],
                "score": float(score),
                "text": self.texts[index],
            }
            for index, score in ranked
        ]

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "texts": self.texts,
            "doc_ids": self.doc_ids,
            "tokenized_texts": self.tokenized_texts,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "avgdl": self.avgdl,
        }
        with self.index_path.open("wb") as file_obj:
            pickle.dump(payload, file_obj)

    def load(self) -> None:
        with self.index_path.open("rb") as file_obj:
            payload = pickle.load(file_obj)
        self.texts = list(payload.get("texts", []))
        self.doc_ids = list(payload.get("doc_ids", []))
        self.tokenized_texts = list(payload.get("tokenized_texts", []))
        self.doc_freqs = list(payload.get("doc_freqs", []))
        self.idf = dict(payload.get("idf", {}))
        self.avgdl = float(payload.get("avgdl", 0.0))
