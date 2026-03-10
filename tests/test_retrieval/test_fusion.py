from app.retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_with_overlap() -> None:
    dense = [("p1", 0.9), ("p2", 0.8), ("p3", 0.7)]
    sparse = [("p2", 12.0), ("p1", 10.0), ("p4", 8.0)]

    results = reciprocal_rank_fusion([dense, sparse], k=60)

    assert results[0][0] in {"p1", "p2"}
    assert dict(results)["p4"] > 0


def test_reciprocal_rank_fusion_without_overlap() -> None:
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("c", 1.0), ("d", 0.7)]

    results = reciprocal_rank_fusion([dense, sparse], k=10)
    ids = [doc_id for doc_id, _ in results]

    # a and c share the same RRF score (rank-1 in each list); both must outrank b and d
    assert set(ids[:2]) == {"a", "c"}
    assert set(ids[2:]) == {"b", "d"}
