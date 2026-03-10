from __future__ import annotations

import re

import pandas as pd
from openai import OpenAI

from app.core.logger import get_logger


logger = get_logger(__name__)

JUDGE_PROMPT = """你是论文问答系统的评测器。
请根据问题、标准答案和预测答案，分别给出：
- faithfulness: 1-5 分，衡量预测答案是否忠于标准答案与事实
- relevance: 1-5 分，衡量预测答案是否回答了问题重点

只允许输出以下格式：
faithfulness: X
relevance: X

问题:
{question}

标准答案:
{gold_answer}

预测答案:
{predicted_answer}
"""


class LLMJudge:
    _score_pattern = re.compile(
        r"(faithfulness|relevance)\s*:\s*([0-5](?:\.\d+)?)",
        re.IGNORECASE,
    )

    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4") -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def score_one(self, question: str, predicted: str, gold: str) -> dict[str, float]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "你是严格的评测助手，只输出分数，不输出解释。",
                },
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        gold_answer=gold,
                        predicted_answer=predicted,
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or ""
        return self._parse_scores(content)

    def score_batch(self, results: list[dict[str, str]]) -> pd.DataFrame:
        rows: list[dict[str, str | float]] = []
        for item in results:
            scores = self.score_one(
                item["question"],
                item["predicted"],
                item["gold"],
            )
            rows.append(
                {
                    "question": item["question"],
                    "faithfulness": scores["faithfulness"],
                    "relevance": scores["relevance"],
                }
            )
        return pd.DataFrame(rows, columns=["question", "faithfulness", "relevance"])

    def _parse_scores(self, content: str) -> dict[str, float]:
        scores = {"faithfulness": 0.0, "relevance": 0.0}
        for match in self._score_pattern.finditer(content):
            metric = match.group(1).lower()
            value = float(match.group(2))
            scores[metric] = value

        if scores["faithfulness"] == 0.0 or scores["relevance"] == 0.0:
            logger.warning("Unexpected judge output format: %s", content)

        return scores
