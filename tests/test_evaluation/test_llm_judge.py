import pandas as pd

from app.evaluation.llm_judge import LLMJudge


class StubJudge(LLMJudge):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.model = "stub"
        self.client = None  # type: ignore[assignment]

    def score_one(self, question: str, predicted: str, gold: str) -> dict[str, float]:
        output = self.outputs[self.calls]
        self.calls += 1
        return self._parse_scores(output)


def test_llm_judge_parses_scores() -> None:
    judge = StubJudge(["faithfulness: 4\nrelevance: 5"])

    scores = judge.score_one("q", "pred", "gold")

    assert scores == {"faithfulness": 4.0, "relevance": 5.0}


def test_llm_judge_returns_zero_for_invalid_format() -> None:
    judge = StubJudge(["not a valid response"])

    scores = judge.score_one("q", "pred", "gold")

    assert scores == {"faithfulness": 0.0, "relevance": 0.0}


def test_llm_judge_score_batch_returns_dataframe() -> None:
    judge = StubJudge(
        [
            "faithfulness: 5\nrelevance: 4",
            "faithfulness: 3\nrelevance: 2",
        ]
    )

    df = judge.score_batch(
        [
            {"question": "q1", "predicted": "p1", "gold": "g1"},
            {"question": "q2", "predicted": "p2", "gold": "g2"},
        ]
    )

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict(orient="records") == [
        {"question": "q1", "faithfulness": 5.0, "relevance": 4.0},
        {"question": "q2", "faithfulness": 3.0, "relevance": 2.0},
    ]
