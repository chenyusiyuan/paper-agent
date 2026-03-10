from __future__ import annotations

from app.core.schemas import AgentState, IntentType
from app.generation.answer_builder import AnswerBuilder


class IntentRouter:
    def __init__(self, answer_builder: AnswerBuilder) -> None:
        self.answer_builder = answer_builder

    def classify(self, query: str, state: AgentState) -> IntentType:
        intent = self.answer_builder.classify_intent(query, state)
        if state.current_paper_id is None and intent in {
            IntentType.PAPER_READING,
            IntentType.SECTION_QA,
        }:
            return IntentType.PAPER_SEARCH
        return intent
