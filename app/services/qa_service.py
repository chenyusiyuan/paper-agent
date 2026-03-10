from __future__ import annotations

from app.agent.orchestrator import Orchestrator
from app.agent.state_manager import StateManager
from app.core.schemas import AgentResponse, AgentState, AgentStateEnum


class QAService:
    def __init__(self, orchestrator: Orchestrator, state_manager: StateManager) -> None:
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.sessions: dict[str, AgentState] = {}

    def chat(self, query: str, session_id: str) -> AgentResponse:
        state = self.sessions.get(session_id, AgentState())
        response = self.orchestrator.run(query, state)
        next_state = (
            self.state_manager.reset_to_routing(response.updated_state)
            if response.updated_state.current_state == AgentStateEnum.ANSWERING
            else response.updated_state
        )
        self.sessions[session_id] = next_state
        return response

    def lock_paper(self, session_id: str, paper_id: str) -> AgentState:
        state = self.sessions.get(session_id, AgentState())
        locked_state = self.state_manager.lock_paper(state, paper_id)
        self.sessions[session_id] = locked_state
        return locked_state

    def get_state(self, session_id: str) -> AgentState:
        return self.sessions.get(session_id, AgentState())
