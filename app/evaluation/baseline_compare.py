from __future__ import annotations

import pandas as pd

from app.evaluation.dataset_builder import EvalSample
from app.evaluation.llm_judge import LLMJudge
from app.generation.answer_builder import AnswerBuilder
from app.indexing.section_index import SectionIndex
from app.services.qa_service import QAService


class FlatRAGBaseline:
    def __init__(
        self,
        section_index: SectionIndex,
        answer_builder: AnswerBuilder,
    ) -> None:
        self.section_index = section_index
        self.answer_builder = answer_builder

    def query(self, question: str) -> str:
        ranked_chunk_ids = self.section_index.search_dense(question, top_k=5)
        chunks = [
            self.section_index.get_by_id(chunk_id)
            for chunk_id, _ in ranked_chunk_ids
        ]
        context = "\n\n".join(chunk.text for chunk in chunks if chunk is not None)
        user_prompt = f"根据以下内容回答问题: {context}\n问题: {question}"
        return self.answer_builder._call_llm(
            "你是论文问答基线系统，请仅根据给定内容回答问题。",
            user_prompt,
        )


class ABComparison:
    def __init__(
        self,
        qa_service: QAService,
        baseline: FlatRAGBaseline,
        judge: LLMJudge,
    ) -> None:
        self.qa_service = qa_service
        self.baseline = baseline
        self.judge = judge

    def run(self, evalset: list[EvalSample]) -> pd.DataFrame:
        rows: list[dict[str, str | float]] = []
        for index, sample in enumerate(evalset):
            agent_answer = self.qa_service.chat(
                sample.question,
                session_id=f"eval_{index}",
            ).answer
            baseline_answer = self.baseline.query(sample.question)
            agent_scores = self.judge.score_one(
                sample.question,
                agent_answer,
                sample.gold_answer,
            )
            baseline_scores = self.judge.score_one(
                sample.question,
                baseline_answer,
                sample.gold_answer,
            )
            rows.append(
                {
                    "question": sample.question,
                    "intent": sample.intent_type,
                    "agent_faith": agent_scores["faithfulness"],
                    "agent_rel": agent_scores["relevance"],
                    "baseline_faith": baseline_scores["faithfulness"],
                    "baseline_rel": baseline_scores["relevance"],
                    "delta_faith": agent_scores["faithfulness"] - baseline_scores["faithfulness"],
                    "delta_rel": agent_scores["relevance"] - baseline_scores["relevance"],
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "question",
                "intent",
                "agent_faith",
                "agent_rel",
                "baseline_faith",
                "baseline_rel",
                "delta_faith",
                "delta_rel",
            ],
        )

    def summary(self, df: pd.DataFrame) -> dict[str, dict[str, float] | dict[str, dict[str, float]]]:
        metric_columns = [
            "agent_faith",
            "agent_rel",
            "baseline_faith",
            "baseline_rel",
            "delta_faith",
            "delta_rel",
        ]
        by_intent: dict[str, dict[str, float]] = {}
        if not df.empty:
            grouped = df.groupby("intent")[metric_columns].mean()
            for intent, row in grouped.iterrows():
                by_intent[str(intent)] = {
                    column: float(row[column])
                    for column in metric_columns
                }

        overall = {
            column: float(df[column].mean()) if not df.empty else 0.0
            for column in metric_columns
        }
        return {"by_intent": by_intent, "overall": overall}
