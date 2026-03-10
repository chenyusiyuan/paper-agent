from __future__ import annotations

from uuid import uuid4

import gradio as gr

from app.api.deps import get_qa_service
from app.core.schemas import AgentResponse, AgentState


def submit_message(
    query: str,
    history: list[tuple[str, str]],
    session_id: str,
) -> tuple[str, list[tuple[str, str]], dict[str, str | list[str] | None], dict[str, object]]:
    qa_service = get_qa_service()
    response = qa_service.chat(query, session_id)
    next_history = (history or []) + [(query, _format_response(response))]
    state = qa_service.get_state(session_id)
    details = _build_details(response)
    return "", next_history, _build_state_payload(state), details


def lock_selected_paper(
    paper_id: str,
    session_id: str,
) -> tuple[str, dict[str, str | list[str] | None]]:
    qa_service = get_qa_service()
    if not paper_id.strip():
        return "", _build_state_payload(qa_service.get_state(session_id))
    state = qa_service.lock_paper(session_id, paper_id.strip())
    return "", _build_state_payload(state)


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="Paper Agent Demo") as demo:
        session_state = gr.State(str(uuid4()))
        with gr.Row():
            chatbot = gr.Chatbot(label="Paper Agent")
            with gr.Column():
                state_view = gr.JSON(label="Current State")
                details_view = gr.JSON(label="Candidates / Evidences")
                lock_input = gr.Textbox(label="Lock Paper ID", placeholder="paper_id")
                lock_button = gr.Button("Lock Paper")
        with gr.Row():
            query_input = gr.Textbox(label="Query", scale=8)
            send_button = gr.Button("Send", scale=1)

        send_button.click(
            submit_message,
            inputs=[query_input, chatbot, session_state],
            outputs=[query_input, chatbot, state_view, details_view],
        )
        query_input.submit(
            submit_message,
            inputs=[query_input, chatbot, session_state],
            outputs=[query_input, chatbot, state_view, details_view],
        )
        lock_button.click(
            lock_selected_paper,
            inputs=[lock_input, session_state],
            outputs=[lock_input, state_view],
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


def _build_details(response: AgentResponse) -> dict[str, object]:
    candidates = []
    if response.candidate_papers:
        candidates = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
            }
            for paper in response.candidate_papers
        ]
    evidences = [
        {
            "paper_id": evidence.paper_id,
            "section_type": evidence.section_type,
            "section_title": evidence.section_title,
            "text": evidence.text,
        }
        for evidence in response.evidences
    ]
    return {"candidate_papers": candidates, "evidences": evidences}


if __name__ == "__main__":
    create_demo().launch()
