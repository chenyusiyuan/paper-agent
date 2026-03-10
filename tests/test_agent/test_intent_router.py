from app.agent.intent_router import IntentRouter
from app.core.schemas import AgentState, IntentType


class StubAnswerBuilder:
    def __init__(self, intent: IntentType) -> None:
        self.intent = intent

    def classify_intent(self, query: str, state: AgentState) -> IntentType:
        return self.intent


def test_intent_router_returns_predicted_paper_search() -> None:
    router = IntentRouter(StubAnswerBuilder(IntentType.PAPER_SEARCH))  # type: ignore[arg-type]

    intent = router.classify("找几篇 RAG 论文", AgentState())

    assert intent == IntentType.PAPER_SEARCH


def test_intent_router_forces_search_when_no_locked_paper() -> None:
    router = IntentRouter(StubAnswerBuilder(IntentType.PAPER_READING))  # type: ignore[arg-type]

    intent = router.classify("总结这篇论文", AgentState(current_paper_id=None))

    assert intent == IntentType.PAPER_SEARCH


def test_intent_router_keeps_search_for_fallback_scenario() -> None:
    router = IntentRouter(StubAnswerBuilder(IntentType.PAPER_SEARCH))  # type: ignore[arg-type]
    state = AgentState(current_paper_id="paper-1", last_intent=IntentType.SECTION_QA)

    intent = router.classify("换一篇讲多模态检索的论文", state)

    assert intent == IntentType.PAPER_SEARCH
