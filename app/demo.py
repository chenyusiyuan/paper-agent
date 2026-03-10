from __future__ import annotations

import html
import time
from uuid import uuid4

import gradio as gr

from app.api.deps import get_qa_service
from app.core.schemas import AgentResponse, AgentState


DEMO_CSS = """
.gradio-container {
    background: #f5f7fb;
}

.app-shell {
    max-width: 1500px;
    margin: 0 auto;
}

.chat-panel, .sidebar-panel {
    background: #ffffff;
    border: 1px solid #e6eaf2;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}

.chat-panel {
    padding: 18px;
}

.sidebar-panel {
    padding: 14px;
}

.chat-title h1 {
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
}

.chat-title p {
    margin: 6px 0 0;
    color: #475569;
    font-size: 15px;
}

.compact-markdown p {
    margin: 0;
}

.candidate-card {
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}

.candidate-card h4 {
    margin: 0 0 8px;
    font-size: 16px;
    color: #0f172a;
}

.candidate-meta {
    color: #475569;
    font-size: 13px;
    margin-bottom: 8px;
}

.candidate-id {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: #e2ecff;
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.candidate-empty {
    color: #64748b;
    font-size: 14px;
}
"""

MAX_CANDIDATE_CARDS = 5


def submit_message(
    query: str,
    history: list[dict[str, str]],
    session_id: str,
    current_state: dict[str, str | list[str] | None],
    candidate_items: list[dict[str, object]],
    current_evidences: list[dict[str, object]],
) -> tuple[
    str,
    list[dict[str, str]],
    dict[str, str | list[str] | None],
    list[dict[str, object]],
    list[dict[str, object]],
    object,
] | tuple[object, ...]:
    qa_service = get_qa_service()
    base_history = (history or []) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "正在思考中..."},
    ]
    yield (
        "",
        base_history,
        current_state,
        candidate_items,
        current_evidences,
        gr.update(visible=not candidate_items),
        *_build_candidate_card_updates(candidate_items),
    )

    response = qa_service.chat(query, session_id)
    state = qa_service.get_state(session_id)
    next_history = (history or []) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": ""},
    ]
    formatted_response = _format_response(response)
    next_candidate_items = _build_candidate_items(response)
    evidences = _build_evidences_payload(response)

    for index in range(1, len(formatted_response) + 1):
        next_history[-1]["content"] = formatted_response[:index]
        yield (
            "",
            next_history,
            _build_state_payload(state),
            next_candidate_items,
            evidences,
            gr.update(visible=not next_candidate_items),
            *_build_candidate_card_updates(next_candidate_items),
        )
        if index < len(formatted_response):
            time.sleep(0.01)


def lock_selected_paper(
    paper_id: str,
    session_id: str,
) -> tuple[str, dict[str, str | list[str] | None]]:
    qa_service = get_qa_service()
    if not paper_id.strip():
        return "", _build_state_payload(qa_service.get_state(session_id))
    state = qa_service.lock_paper(session_id, paper_id.strip())
    return "", _build_state_payload(state)


def lock_candidate(
    index: int,
    history: list[dict[str, str]],
    session_id: str,
    candidate_items: list[dict[str, object]],
) -> tuple[list[dict[str, str]], dict[str, str | list[str] | None]]:
    qa_service = get_qa_service()
    if index >= len(candidate_items):
        return history or [], _build_state_payload(qa_service.get_state(session_id))

    candidate = candidate_items[index]
    paper_id = str(candidate["paper_id"])
    state = qa_service.lock_paper(session_id, paper_id)
    next_history = (history or []) + [
        {
            "role": "assistant",
            "content": (
                f"已锁定论文：{candidate['title']} ({paper_id})。\n\n"
                "接下来可以继续问这篇论文的核心贡献、方法、实验或具体章节问题。"
            ),
        }
    ]
    return next_history, _build_state_payload(state)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Paper Agent Demo", css=DEMO_CSS, fill_height=True) as demo:
        session_state = gr.State(str(uuid4()))
        candidate_items_state = gr.State([])
        with gr.Row(elem_classes="app-shell", equal_height=True):
            with gr.Column(scale=8, elem_classes="chat-panel"):
                gr.Markdown(
                    """
                    <div class="chat-title">
                      <h1>Paper Agent</h1>
                      <p>搜索论文、锁定目标论文、继续做精读与章节问答。</p>
                    </div>
                    """,
                    elem_classes="compact-markdown",
                )
                chatbot = gr.Chatbot(
                    label=None,
                    height=740,
                    layout="bubble",
                    placeholder="在这里开始一轮论文搜索或阅读对话。",
                )
                with gr.Row():
                    query_input = gr.Textbox(
                        label=None,
                        placeholder="例如：检索一下 RAG 相关论文，或总结 paper-020 的方法和实验结论",
                        lines=3,
                        scale=8,
                    )
                    send_button = gr.Button("Send", scale=1, variant="primary")
            with gr.Column(scale=4, elem_classes="sidebar-panel"):
                gr.Markdown("### Session")
                state_view = gr.JSON(label="Current State")
                with gr.Accordion("Candidate Papers", open=True):
                    candidate_card_markdowns: list[gr.Markdown] = []
                    candidate_lock_buttons: list[gr.Button] = []
                    for _ in range(MAX_CANDIDATE_CARDS):
                        candidate_card_markdowns.append(
                            gr.Markdown(
                                value="",
                                visible=False,
                                elem_classes="compact-markdown",
                            )
                        )
                        candidate_lock_buttons.append(
                            gr.Button(
                                "锁定这篇论文",
                                visible=False,
                                variant="secondary",
                            )
                        )
                    empty_candidates_view = gr.Markdown(
                        '<div class="candidate-empty">暂无候选论文。</div>'
                    )
                with gr.Accordion("Evidences", open=True):
                    evidences_view = gr.JSON(label=None)
                gr.Markdown("### Lock Paper")
                lock_input = gr.Textbox(label=None, placeholder="输入 paper_id，例如 paper-020")
                lock_button = gr.Button("Lock Paper")

        send_button.click(
            submit_message,
            inputs=[
                query_input,
                chatbot,
                session_state,
                state_view,
                candidate_items_state,
                evidences_view,
            ],
            outputs=[
                query_input,
                chatbot,
                state_view,
                candidate_items_state,
                evidences_view,
                empty_candidates_view,
                *candidate_card_markdowns,
                *candidate_lock_buttons,
            ],
        )
        query_input.submit(
            submit_message,
            inputs=[
                query_input,
                chatbot,
                session_state,
                state_view,
                candidate_items_state,
                evidences_view,
            ],
            outputs=[
                query_input,
                chatbot,
                state_view,
                candidate_items_state,
                evidences_view,
                empty_candidates_view,
                *candidate_card_markdowns,
                *candidate_lock_buttons,
            ],
        )
        lock_button.click(
            lock_selected_paper,
            inputs=[lock_input, session_state],
            outputs=[lock_input, state_view],
        )
        for index, candidate_lock_button in enumerate(candidate_lock_buttons):
            candidate_lock_button.click(
                lambda history, session_id, candidate_items, idx=index: lock_candidate(
                    idx,
                    history,
                    session_id,
                    candidate_items,
                ),
                inputs=[chatbot, session_state, candidate_items_state],
                outputs=[chatbot, state_view],
            )

    return demo


def _format_response(response: AgentResponse) -> str:
    evidence_lines = [
        f"- [{evidence.section_type}] {evidence.section_title}"
        for evidence in response.evidences
    ]
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- 无"
    return f"意图: {response.intent.value}\n\n{response.answer}\n\n来源:\n{evidence_block}"


def _build_state_payload(state: AgentState) -> dict[str, str | list[str] | None]:
    return {
        "current_state": state.current_state.value,
        "current_paper_id": state.current_paper_id,
        "candidate_papers": state.candidate_papers,
        "current_focus_section": state.current_focus_section,
        "last_intent": state.last_intent.value if state.last_intent else None,
    }


def _build_candidate_items(response: AgentResponse) -> list[dict[str, object]]:
    if not response.candidate_papers:
        return []
    return [
        {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
        }
        for paper in response.candidate_papers
    ]


def _render_candidate_card(candidate: dict[str, object], index: int) -> str:
    authors = ", ".join(str(author) for author in list(candidate["authors"])[:4])
    if len(list(candidate["authors"])) > 4:
        authors = f"{authors}, ..."
    year = candidate["year"] if candidate["year"] is not None else "未知"
    return (
        f"""
<div class="candidate-card">
  <div class="candidate-id">#{index} · {html.escape(str(candidate["paper_id"]))}</div>
  <h4>{html.escape(str(candidate["title"]))}</h4>
  <div class="candidate-meta">作者：{html.escape(authors or "未知")} | 年份：{html.escape(str(year))}</div>
</div>
""".strip()
    )


def _build_candidate_card_updates(candidate_items: list[dict[str, object]]) -> list[object]:
    updates: list[object] = []
    for index in range(MAX_CANDIDATE_CARDS):
        if index < len(candidate_items):
            updates.append(
                gr.update(
                    value=_render_candidate_card(candidate_items[index], index + 1),
                    visible=True,
                )
            )
        else:
            updates.append(gr.update(value="", visible=False))
    for index in range(MAX_CANDIDATE_CARDS):
        updates.append(gr.update(visible=index < len(candidate_items)))
    return updates


def _build_evidences_payload(response: AgentResponse) -> list[dict[str, object]]:
    return [
        {
            "paper_id": evidence.paper_id,
            "section_type": evidence.section_type,
            "section_title": evidence.section_title,
            "text": evidence.text,
        }
        for evidence in response.evidences
    ]


if __name__ == "__main__":
    create_demo().launch(server_name="0.0.0.0", server_port=7860)
