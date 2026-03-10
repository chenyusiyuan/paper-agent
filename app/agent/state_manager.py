from __future__ import annotations

from dataclasses import replace

from app.core.schemas import AgentState, AgentStateEnum, IntentType


class StateManager:
    def transition(self, state: AgentState, intent: IntentType, query: str) -> AgentState:
        if intent == IntentType.PAPER_SEARCH:
            return replace(
                state,
                current_state=AgentStateEnum.SEARCHING,
                candidate_papers=[],
                current_paper_id=None,
                current_focus_section=None,
                last_intent=intent,
            )
        if intent == IntentType.PAPER_READING:
            next_state = (
                AgentStateEnum.LOCKED
                if state.current_paper_id is not None
                else AgentStateEnum.SEARCHING
            )
            return replace(
                state,
                current_state=next_state,
                current_focus_section=None,
                last_intent=intent,
            )
        return replace(
            state,
            current_state=(
                AgentStateEnum.READING
                if state.current_paper_id is not None
                else AgentStateEnum.SEARCHING
            ),
            last_intent=intent,
        )

    def lock_paper(self, state: AgentState, paper_id: str) -> AgentState:
        return replace(
            state,
            current_paper_id=paper_id,
            current_state=AgentStateEnum.LOCKED,
            current_focus_section=None,
        )

    def to_answering(self, state: AgentState) -> AgentState:
        return replace(state, current_state=AgentStateEnum.ANSWERING)

    def reset_to_routing(self, state: AgentState) -> AgentState:
        return replace(
            state,
            current_state=AgentStateEnum.ROUTING,
            candidate_papers=[],
            current_focus_section=None,
        )
