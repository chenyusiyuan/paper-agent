from pathlib import Path

from app.evaluation.dataset_builder import EvalSample, load_evalset, save_evalset


def test_load_and_save_evalset_round_trip(tmp_path: Path) -> None:
    eval_path = tmp_path / "evalset.jsonl"
    samples = [
        EvalSample(
            question="q1",
            intent_type="paper_search",
            gold_paper_ids=["p1"],
            gold_section_ids=["s1"],
            gold_answer="a1",
        ),
        EvalSample(
            question="q2",
            intent_type="section_qa",
            gold_paper_ids=["p2"],
            gold_section_ids=["s2"],
            gold_answer="a2",
        ),
    ]

    save_evalset(samples, str(eval_path))
    loaded = load_evalset(str(eval_path))

    assert loaded == samples


def test_load_evalset_returns_empty_when_missing(tmp_path: Path) -> None:
    loaded = load_evalset(str(tmp_path / "missing.jsonl"))

    assert loaded == []
