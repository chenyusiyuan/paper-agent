from __future__ import annotations

from pathlib import Path

from app.api.deps import get_paper_index, get_section_index, get_settings
from app.core.logger import get_logger
from app.evaluation import EvalSample, load_evalset, save_evalset


logger = get_logger(__name__)
VALID_INTENTS = ["paper_search", "paper_reading", "section_qa"]


def main() -> None:
    settings = get_settings()
    paper_index = get_paper_index()
    section_index = get_section_index()
    eval_path = Path(settings.data_dir) / "eval" / "evalset.jsonl"
    samples = load_evalset(str(eval_path))

    papers = sorted(paper_index.metadata_store.values(), key=lambda paper: paper.paper_id)
    print("已入库论文:")
    for paper in papers:
        print(f"- {paper.paper_id}: {paper.title}")

    while True:
        question = input("Question (直接回车结束): ").strip()
        if not question:
            break

        intent_type = _prompt_intent()
        gold_paper_ids = _prompt_list("Gold paper ids (逗号分隔，可为空): ")
        if gold_paper_ids:
            for paper_id in gold_paper_ids:
                chunks = section_index.get_by_paper(paper_id)
                print(f"{paper_id} 可选 section ids:")
                for chunk in chunks[:20]:
                    print(f"- {chunk.chunk_id}: {chunk.section_title}")
        gold_section_ids = _prompt_list("Gold section ids (逗号分隔，可为空): ")
        gold_answer = input("Gold answer: ").strip()

        samples.append(
            EvalSample(
                question=question,
                intent_type=intent_type,
                gold_paper_ids=gold_paper_ids,
                gold_section_ids=gold_section_ids,
                gold_answer=gold_answer,
            )
        )
        save_evalset(samples, str(eval_path))
        logger.info("Saved %s samples to %s", len(samples), eval_path)


def _prompt_intent() -> str:
    while True:
        intent = input(
            f"Intent type ({'/'.join(VALID_INTENTS)}): "
        ).strip()
        if intent in VALID_INTENTS:
            return intent
        print("Invalid intent type.")


def _prompt_list(prompt: str) -> list[str]:
    raw_value = input(prompt).strip()
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
