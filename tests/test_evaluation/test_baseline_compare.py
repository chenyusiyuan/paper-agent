from app.core.schemas import AgentResponse, AgentState, AgentStateEnum, IntentType, SectionChunk
from app.evaluation.baseline_compare import ABComparison, FlatRAGBaseline
from app.evaluation.dataset_builder import EvalSample


def make_chunk(chunk_id: str, text: str) -> SectionChunk:
    return SectionChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        section_type="method",
        section_title="Method",
        section_path="Method",
        text=text,
        page_start=0,
        page_end=0,
        order_in_paper=0,
        level=1,
        parent_chunk_id=None,
        granularity="fine",
    )


class StubSectionIndex:
    def __init__(self) -> None:
        self.chunks = {
            "c1": make_chunk("c1", "chunk one"),
            "c2": make_chunk("c2", "chunk two"),
        }

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        return [("c1", 0.9), ("c2", 0.8)]

    def get_by_id(self, chunk_id: str) -> SectionChunk | None:
        return self.chunks.get(chunk_id)


class StubAnswerBuilder:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return "baseline answer"


class StubQAService:
    def chat(self, question: str, session_id: str) -> AgentResponse:
        return AgentResponse(
            answer="agent answer",
            evidences=[],
            intent=IntentType.SECTION_QA,
            updated_state=AgentState(current_state=AgentStateEnum.ROUTING),
            candidate_papers=None,
        )


class StubJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def score_one(self, question: str, predicted: str, gold: str) -> dict[str, float]:
        self.calls.append((question, predicted, gold))
        if predicted == "agent answer":
            return {"faithfulness": 5.0, "relevance": 4.0}
        return {"faithfulness": 3.0, "relevance": 2.0}



def test_flat_rag_baseline_uses_dense_topk_context() -> None:
    answer_builder = StubAnswerBuilder()
    baseline = FlatRAGBaseline(
        section_index=StubSectionIndex(),  # type: ignore[arg-type]
        answer_builder=answer_builder,  # type: ignore[arg-type]
    )

    answer = baseline.query("what is the method")

    assert answer == "baseline answer"
    assert "chunk one" in answer_builder.prompts[0]
    assert "chunk two" in answer_builder.prompts[0]


def test_ab_comparison_run_and_summary() -> None:
    comparison = ABComparison(
        qa_service=StubQAService(),  # type: ignore[arg-type]
        baseline=FlatRAGBaseline(
            section_index=StubSectionIndex(),  # type: ignore[arg-type]
            answer_builder=StubAnswerBuilder(),  # type: ignore[arg-type]
        ),
        judge=StubJudge(),  # type: ignore[arg-type]
    )
    evalset = [
        EvalSample(
            question="q1",
            intent_type="section_qa",
            gold_paper_ids=["paper-1"],
            gold_section_ids=["c1"],
            gold_answer="gold",
        ),
        EvalSample(
            question="q2",
            intent_type="paper_reading",
            gold_paper_ids=["paper-1"],
            gold_section_ids=["c1"],
            gold_answer="gold",
        ),
    ]

    df = comparison.run(evalset)
    summary = comparison.summary(df)

    assert list(df["delta_faith"]) == [2.0, 2.0]
    assert summary["overall"]["delta_rel"] == 2.0
    assert summary["by_intent"]["paper_reading"]["agent_faith"] == 5.0
