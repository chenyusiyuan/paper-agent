from fastapi.testclient import TestClient

from app.api.deps import get_ingest_service, get_paper_index, get_qa_service
from app.core.schemas import (
    AgentResponse,
    AgentState,
    AgentStateEnum,
    Evidence,
    IntentType,
    PaperMetadata,
)
from app.main import app


class StubQAService:
    def chat(self, query: str, session_id: str) -> AgentResponse:
        return AgentResponse(
            answer=f"answer for {query}",
            evidences=[
                Evidence(
                    text="method text",
                    paper_id="paper-1",
                    section_type="method",
                    section_title="Method",
                    page=1,
                )
            ],
            intent=IntentType.SECTION_QA,
            updated_state=AgentState(
                current_state=AgentStateEnum.ROUTING,
                candidate_papers=[],
                current_paper_id="paper-1",
                current_focus_section="method",
                last_intent=IntentType.SECTION_QA,
            ),
            candidate_papers=None,
        )

    def lock_paper(self, session_id: str, paper_id: str) -> AgentState:
        return AgentState(
            current_state=AgentStateEnum.LOCKED,
            current_paper_id=paper_id,
            last_intent=IntentType.PAPER_SEARCH,
        )

    def get_state(self, session_id: str) -> AgentState:
        return AgentState(current_state=AgentStateEnum.ROUTING)


class StubIngestService:
    def ingest(self, pdf_dir: str) -> dict[str, int | list[dict[str, str]]]:
        return {
            "total": 2,
            "success": 2,
            "failed": 0,
            "errors": [],
        }


class StubPaperIndex:
    def __init__(self) -> None:
        self.metadata_store = {
            "paper-1": PaperMetadata(
                paper_id="paper-1",
                title="Paper One",
                authors=["A"],
                year=2024,
                venue="ACL",
                abstract="Paper one abstract",
                keywords=["rag"],
                section_titles=["Abstract"],
            ),
            "paper-2": PaperMetadata(
                paper_id="paper-2",
                title="Paper Two",
                authors=["B"],
                year=2023,
                venue="NeurIPS",
                abstract="Paper two abstract",
                keywords=["agent"],
                section_titles=["Method"],
            ),
        }

    def get_by_id(self, paper_id: str) -> PaperMetadata | None:
        return self.metadata_store.get(paper_id)


def test_chat_route_returns_agent_response() -> None:
    app.dependency_overrides[get_qa_service] = lambda: StubQAService()
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"query": "训练细节是什么", "session_id": "s1"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["answer"] == "answer for 训练细节是什么"
        assert payload["intent"] == "section_qa"
        assert payload["updated_state"]["current_paper_id"] == "paper-1"
        assert payload["evidences"][0]["section_type"] == "method"
    app.dependency_overrides.clear()


def test_ingest_route_returns_summary() -> None:
    app.dependency_overrides[get_ingest_service] = lambda: StubIngestService()
    with TestClient(app) as client:
        response = client.post("/api/ingest", json={"pdf_dir": "data/raw_pdfs"})

        assert response.status_code == 200
        assert response.json() == {
            "total": 2,
            "success": 2,
            "failed": 0,
            "errors": [],
        }
    app.dependency_overrides.clear()


def test_list_papers_route_returns_metadata() -> None:
    app.dependency_overrides[get_paper_index] = lambda: StubPaperIndex()
    with TestClient(app) as client:
        response = client.get("/api/papers")

        assert response.status_code == 200
        payload = response.json()
        assert [paper["paper_id"] for paper in payload] == ["paper-1", "paper-2"]
        assert payload[0]["title"] == "Paper One"
    app.dependency_overrides.clear()
