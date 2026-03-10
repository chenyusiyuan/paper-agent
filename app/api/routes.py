from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import (
    get_ingest_service,
    get_paper_index,
    get_qa_service,
    reset_runtime_dependencies,
)
from app.core.schemas import AgentResponse, AgentState, Evidence, PaperMetadata
from app.indexing.paper_index import PaperIndex
from app.services import IngestService, QAService


router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    query: str
    session_id: str


class LockPaperRequest(BaseModel):
    session_id: str
    paper_id: str


class IngestRequest(BaseModel):
    pdf_dir: str


class PaperMetadataModel(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str
    keywords: list[str]
    section_titles: list[str]


class AgentStateModel(BaseModel):
    current_state: str
    candidate_papers: list[str]
    current_paper_id: str | None
    current_focus_section: str | None
    last_intent: str | None


class EvidenceModel(BaseModel):
    text: str
    paper_id: str
    section_type: str
    section_title: str
    page: int | None


class AgentResponseModel(BaseModel):
    answer: str
    evidences: list[EvidenceModel]
    intent: str
    updated_state: AgentStateModel
    candidate_papers: list[PaperMetadataModel] | None = None


class IngestResultModel(BaseModel):
    total: int
    success: int
    failed: int
    errors: list[dict[str, str]]


@router.post("/chat", response_model=AgentResponseModel)
def chat(
    request: ChatRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> AgentResponseModel:
    return _to_agent_response_model(qa_service.chat(request.query, request.session_id))


@router.post("/chat/lock", response_model=AgentStateModel)
def lock_paper(
    request: LockPaperRequest,
    qa_service: QAService = Depends(get_qa_service),
) -> AgentStateModel:
    return _to_agent_state_model(qa_service.lock_paper(request.session_id, request.paper_id))


@router.post("/ingest", response_model=IngestResultModel)
def ingest(
    request: IngestRequest,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestResultModel:
    result = ingest_service.ingest(request.pdf_dir)
    reset_runtime_dependencies()
    return IngestResultModel(**result)


@router.get("/papers", response_model=list[PaperMetadataModel])
def list_papers(
    paper_index: PaperIndex = Depends(get_paper_index),
) -> list[PaperMetadataModel]:
    papers = sorted(paper_index.metadata_store.values(), key=lambda item: item.paper_id)
    return [_to_paper_metadata_model(paper) for paper in papers]


@router.get("/papers/{paper_id}", response_model=PaperMetadataModel)
def get_paper(
    paper_id: str,
    paper_index: PaperIndex = Depends(get_paper_index),
) -> PaperMetadataModel:
    paper = paper_index.get_by_id(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return _to_paper_metadata_model(paper)


@router.get("/session/{session_id}", response_model=AgentStateModel)
def get_session(
    session_id: str,
    qa_service: QAService = Depends(get_qa_service),
) -> AgentStateModel:
    return _to_agent_state_model(qa_service.get_state(session_id))


def _to_paper_metadata_model(metadata: PaperMetadata) -> PaperMetadataModel:
    return PaperMetadataModel(
        paper_id=metadata.paper_id,
        title=metadata.title,
        authors=metadata.authors,
        year=metadata.year,
        venue=metadata.venue,
        abstract=metadata.abstract,
        keywords=metadata.keywords,
        section_titles=metadata.section_titles,
    )


def _to_agent_state_model(state: AgentState) -> AgentStateModel:
    return AgentStateModel(
        current_state=state.current_state.value,
        candidate_papers=state.candidate_papers,
        current_paper_id=state.current_paper_id,
        current_focus_section=state.current_focus_section,
        last_intent=state.last_intent.value if state.last_intent else None,
    )


def _to_evidence_model(evidence: Evidence) -> EvidenceModel:
    return EvidenceModel(
        text=evidence.text,
        paper_id=evidence.paper_id,
        section_type=evidence.section_type,
        section_title=evidence.section_title,
        page=evidence.page,
    )


def _to_agent_response_model(response: AgentResponse) -> AgentResponseModel:
    candidates = None
    if response.candidate_papers is not None:
        candidates = [_to_paper_metadata_model(paper) for paper in response.candidate_papers]
    return AgentResponseModel(
        answer=response.answer,
        evidences=[_to_evidence_model(item) for item in response.evidences],
        intent=response.intent.value,
        updated_state=_to_agent_state_model(response.updated_state),
        candidate_papers=candidates,
    )
