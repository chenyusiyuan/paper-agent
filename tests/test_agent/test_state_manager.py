from app.agent.state_manager import StateManager
from app.core.schemas import AgentState, AgentStateEnum, IntentType


def test_state_manager_search_transition_clears_locked_paper() -> None:
    manager = StateManager()
    original = AgentState(
        current_state=AgentStateEnum.LOCKED,
        candidate_papers=["paper-1"],
        current_paper_id="paper-1",
        current_focus_section="method",
        last_intent=IntentType.SECTION_QA,
    )

    updated = manager.transition(original, IntentType.PAPER_SEARCH, "new query")

    assert updated.current_state == AgentStateEnum.SEARCHING
    assert updated.current_paper_id is None
    assert updated.candidate_papers == []
    assert updated.current_focus_section is None
    assert updated.last_intent == IntentType.PAPER_SEARCH
    assert original.current_paper_id == "paper-1"


def test_state_manager_paper_reading_with_locked_paper_stays_locked() -> None:
    manager = StateManager()
    original = AgentState(current_paper_id="paper-1")

    updated = manager.transition(original, IntentType.PAPER_READING, "summary")

    assert updated.current_state == AgentStateEnum.LOCKED
    assert updated.current_paper_id == "paper-1"
    assert updated.last_intent == IntentType.PAPER_READING


def test_state_manager_paper_reading_without_paper_goes_searching() -> None:
    manager = StateManager()

    updated = manager.transition(AgentState(), IntentType.PAPER_READING, "summary")

    assert updated.current_state == AgentStateEnum.SEARCHING
    assert updated.current_paper_id is None


def test_state_manager_section_qa_with_locked_paper_goes_reading() -> None:
    manager = StateManager()

    updated = manager.transition(
        AgentState(current_paper_id="paper-1"),
        IntentType.SECTION_QA,
        "细节问题",
    )

    assert updated.current_state == AgentStateEnum.READING
    assert updated.last_intent == IntentType.SECTION_QA


def test_state_manager_section_qa_without_paper_goes_searching() -> None:
    manager = StateManager()

    updated = manager.transition(AgentState(), IntentType.SECTION_QA, "细节问题")

    assert updated.current_state == AgentStateEnum.SEARCHING


def test_state_manager_helpers_return_new_state_instances() -> None:
    manager = StateManager()
    original = AgentState(current_state=AgentStateEnum.SEARCHING)

    locked = manager.lock_paper(original, "paper-1")
    answering = manager.to_answering(locked)
    routed = manager.reset_to_routing(answering)

    assert locked is not original
    assert locked.current_paper_id == "paper-1"
    assert locked.current_state == AgentStateEnum.LOCKED
    assert answering.current_state == AgentStateEnum.ANSWERING
    assert routed.current_state == AgentStateEnum.ROUTING
    assert routed.current_paper_id == "paper-1"
    assert routed.candidate_papers == []
    assert routed.current_focus_section is None
