from app.agent.orchestrator import Orchestrator
from app.agent.state_manager import StateManager
from app.core.schemas import AgentState, AgentStateEnum, PaperMetadata, SectionChunk
from app.core.schemas import Evidence, IntentType


def make_paper(paper_id: str, title: str) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=["Author A"],
        year=2024,
        venue="ACL",
        abstract=f"{title} abstract",
        keywords=["rag"],
        section_titles=["Abstract", "Method", "Experiment", "Conclusion"],
    )


def make_chunk(chunk_id: str, section_type: str, text: str, order: int) -> SectionChunk:
    return SectionChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        section_type=section_type,
        section_title=section_type.title(),
        section_path=section_type.title(),
        text=text,
        page_start=order,
        page_end=order,
        order_in_paper=order,
        level=1,
        parent_chunk_id=None,
        granularity="fine",
    )


class StubIntentRouter:
    def __init__(self, intent: IntentType) -> None:
        self.intent = intent

    def classify(self, query: str, state: AgentState) -> IntentType:
        return self.intent


class StubTools:
    def __init__(self) -> None:
        self.candidates = [
            make_paper("paper-1", "Paper One"),
            make_paper("paper-2", "Paper Two"),
        ]
        self.metadata = {"paper-1": self.candidates[0], "paper-2": self.candidates[1]}

    def search_papers(self, query: str, top_k: int = 3) -> list[PaperMetadata]:
        return self.candidates[:top_k]

    def get_paper_metadata(self, paper_id: str) -> PaperMetadata | None:
        return self.metadata.get(paper_id)

    def retrieve_sections(
        self,
        paper_id: str,
        query: str,
        target_sections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[SectionChunk]:
        if target_sections == ["abstract", "conclusion"]:
            return [
                make_chunk("c1", "abstract", "abstract content", 0),
                make_chunk("c2", "conclusion", "conclusion content", 3),
            ]
        if target_sections == ["method", "experiment"]:
            return [
                make_chunk("c3", "method", "method content", 1),
                make_chunk("c2", "conclusion", "conclusion content", 3),
                make_chunk("c4", "experiment", "experiment content", 2),
            ]
        return [
            make_chunk("c5", "method", "training details", 1),
            make_chunk("c6", "experiment", "ablation details", 2),
        ][:top_k]


class StubAnswerBuilder:
    def __init__(self) -> None:
        self.paper_reading_calls: list[tuple[str, list[Evidence], PaperMetadata]] = []
        self.section_qa_calls: list[tuple[str, list[Evidence], PaperMetadata]] = []

    def generate_candidates_summary(
        self,
        candidates: list[PaperMetadata],
        query: str,
    ) -> str:
        return f"candidates:{','.join(candidate.paper_id for candidate in candidates)}"

    def generate_paper_reading(
        self,
        query: str,
        evidences: list[Evidence],
        metadata: PaperMetadata,
    ) -> str:
        self.paper_reading_calls.append((query, evidences, metadata))
        return "paper reading answer"

    def generate_section_qa(
        self,
        query: str,
        evidences: list[Evidence],
        metadata: PaperMetadata,
    ) -> str:
        self.section_qa_calls.append((query, evidences, metadata))
        return "section qa answer"



def test_orchestrator_runs_paper_search_flow() -> None:
    orchestrator = Orchestrator(
        intent_router=StubIntentRouter(IntentType.PAPER_SEARCH),  # type: ignore[arg-type]
        state_manager=StateManager(),
        tools=StubTools(),  # type: ignore[arg-type]
        answer_builder=StubAnswerBuilder(),  # type: ignore[arg-type]
    )

    response = orchestrator.run("找论文", AgentState())

    assert response.intent == IntentType.PAPER_SEARCH
    assert response.answer == "candidates:paper-1,paper-2"
    assert response.evidences == []
    assert response.updated_state.current_state == AgentStateEnum.SEARCHING
    assert response.updated_state.candidate_papers == ["paper-1", "paper-2"]
    assert response.candidate_papers is not None
    assert [paper.paper_id for paper in response.candidate_papers] == ["paper-1", "paper-2"]


def test_orchestrator_runs_paper_reading_flow_and_deduplicates_sections() -> None:
    answer_builder = StubAnswerBuilder()
    orchestrator = Orchestrator(
        intent_router=StubIntentRouter(IntentType.PAPER_READING),  # type: ignore[arg-type]
        state_manager=StateManager(),
        tools=StubTools(),  # type: ignore[arg-type]
        answer_builder=answer_builder,  # type: ignore[arg-type]
    )
    state = AgentState(current_paper_id="paper-1")

    response = orchestrator.run("总结论文", state)

    assert response.intent == IntentType.PAPER_READING
    assert response.answer == "paper reading answer"
    assert response.updated_state.current_state == AgentStateEnum.ANSWERING
    assert response.candidate_papers is None
    assert [evidence.section_type for evidence in response.evidences] == [
        "abstract",
        "conclusion",
        "method",
        "experiment",
    ]
    assert len(answer_builder.paper_reading_calls) == 1
    assert len(answer_builder.paper_reading_calls[0][1]) == 4


def test_orchestrator_runs_section_qa_flow_and_sets_focus_section() -> None:
    answer_builder = StubAnswerBuilder()
    orchestrator = Orchestrator(
        intent_router=StubIntentRouter(IntentType.SECTION_QA),  # type: ignore[arg-type]
        state_manager=StateManager(),
        tools=StubTools(),  # type: ignore[arg-type]
        answer_builder=answer_builder,  # type: ignore[arg-type]
    )
    state = AgentState(current_paper_id="paper-1")

    response = orchestrator.run("训练细节是什么", state)

    assert response.intent == IntentType.SECTION_QA
    assert response.answer == "section qa answer"
    assert response.updated_state.current_state == AgentStateEnum.ANSWERING
    assert response.updated_state.current_focus_section == "method"
    assert [evidence.section_type for evidence in response.evidences] == ["method", "experiment"]
    assert len(answer_builder.section_qa_calls) == 1
