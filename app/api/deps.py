from __future__ import annotations

from functools import lru_cache

from app.agent import AgentTools, IntentRouter, Orchestrator, StateManager
from app.core.config import Settings, get_settings
from app.generation import AnswerBuilder
from app.indexing import PaperIndex, SectionIndex
from app.retrieval.paper_retriever import PaperRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.section_retriever import SectionRetriever
from app.services import IngestService, QAService


@lru_cache(maxsize=1)
def get_paper_index() -> PaperIndex:
    settings = get_settings()
    return PaperIndex(
        settings.data_dir,
        settings.embedding_model,
        embedding_batch_size=settings.embedding_batch_size,
        embedding_max_seq_length=settings.embedding_max_seq_length,
    )


@lru_cache(maxsize=1)
def get_section_index() -> SectionIndex:
    settings = get_settings()
    return SectionIndex(
        settings.data_dir,
        settings.embedding_model,
        embedding_batch_size=settings.embedding_batch_size,
        embedding_max_seq_length=settings.embedding_max_seq_length,
    )


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(settings.reranker_model)


@lru_cache(maxsize=1)
def get_answer_builder() -> AnswerBuilder:
    settings = get_settings()
    return AnswerBuilder(
        settings.llm_api_key,
        settings.llm_base_url,
        settings.llm_model,
    )


@lru_cache(maxsize=1)
def get_state_manager() -> StateManager:
    return StateManager()


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    paper_index = get_paper_index()
    section_index = get_section_index()
    reranker = get_reranker()
    paper_retriever = PaperRetriever(paper_index, reranker)
    section_retriever = SectionRetriever(section_index, reranker)
    tools = AgentTools(paper_retriever, section_retriever, paper_index)
    return Orchestrator(
        intent_router=IntentRouter(get_answer_builder()),
        state_manager=get_state_manager(),
        tools=tools,
        answer_builder=get_answer_builder(),
    )


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    return QAService(get_orchestrator(), get_state_manager())


@lru_cache(maxsize=1)
def get_ingest_service() -> IngestService:
    return IngestService(get_settings())


def reset_runtime_dependencies() -> None:
    get_paper_index.cache_clear()
    get_section_index.cache_clear()
    get_orchestrator.cache_clear()
    get_qa_service.cache_clear()
