from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntentType(str, Enum):
    PAPER_SEARCH = "paper_search"
    PAPER_READING = "paper_reading"
    SECTION_QA = "section_qa"


class AgentStateEnum(str, Enum):
    ROUTING = "routing"
    SEARCHING = "searching"
    LOCKED = "locked"
    READING = "reading"
    ANSWERING = "answering"


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    abstract: str
    keywords: list[str]
    section_titles: list[str]


@dataclass
class SectionChunk:
    chunk_id: str
    paper_id: str
    section_type: str
    section_title: str
    section_path: str
    text: str
    page_start: int
    page_end: int
    order_in_paper: int
    level: int
    parent_chunk_id: str | None
    granularity: str


@dataclass
class AgentState:
    current_state: AgentStateEnum = AgentStateEnum.ROUTING
    candidate_papers: list[str] = field(default_factory=list)
    current_paper_id: str | None = None
    current_focus_section: str | None = None
    last_intent: IntentType | None = None


@dataclass
class Evidence:
    text: str
    paper_id: str
    section_type: str
    section_title: str
    page: int | None = None


@dataclass
class AgentResponse:
    answer: str
    evidences: list[Evidence]
    intent: IntentType
    updated_state: AgentState
    candidate_papers: list[PaperMetadata] | None = None
