from __future__ import annotations

from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        if not passages:
            return []

        sentence_pairs = [[query, passage] for passage in passages]
        scores = self.model.compute_score(sentence_pairs)
        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(
            [(index, float(score)) for index, score in enumerate(scores)],
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]
