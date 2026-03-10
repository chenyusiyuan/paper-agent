from __future__ import annotations

import json
from pathlib import Path

from app.api.deps import (
    get_answer_builder,
    get_qa_service,
    get_section_index,
    get_settings,
)
from app.core.logger import get_logger
from app.core.schemas import AgentState
from app.evaluation import ABComparison, EvalSample, FlatRAGBaseline, LLMJudge, load_evalset
from app.services.qa_service import QAService


logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    evalset = load_evalset()
    qa_service = get_qa_service()
    baseline = FlatRAGBaseline(
        section_index=get_section_index(),
        answer_builder=get_answer_builder(),
    )
    judge = LLMJudge(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
    comparison = ABComparison(qa_service, baseline, judge)
    results_df = comparison.run(evalset)

    eval_dir = Path(settings.data_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    routing_accuracy = _compute_routing_accuracy(qa_service, evalset)
    summary = comparison.summary(results_df)
    summary["routing_accuracy"] = {
        "correct": routing_accuracy["correct"],
        "total": routing_accuracy["total"],
        "accuracy": routing_accuracy["accuracy"],
    }

    logger.info("Saved evaluation results to %s", results_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _compute_routing_accuracy(
    qa_service: QAService,
    evalset: list[EvalSample],
) -> dict[str, float | int]:
    correct = 0
    total = len(evalset)
    for sample in evalset:
        current_paper_id = None
        if sample.intent_type != "paper_search" and sample.gold_paper_ids:
            current_paper_id = sample.gold_paper_ids[0]
        state = AgentState(current_paper_id=current_paper_id)
        predicted = qa_service.orchestrator.intent_router.classify(sample.question, state)
        if predicted.value == sample.intent_type:
            correct += 1
    accuracy = float(correct / total) if total else 0.0
    return {"correct": correct, "total": total, "accuracy": accuracy}


if __name__ == "__main__":
    main()
