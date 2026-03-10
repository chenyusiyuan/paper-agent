from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvalSample:
    question: str
    intent_type: str
    gold_paper_ids: list[str]
    gold_section_ids: list[str]
    gold_answer: str


def load_evalset(path: str = "data/eval/evalset.jsonl") -> list[EvalSample]:
    eval_path = Path(path)
    if not eval_path.exists():
        return []

    samples: list[EvalSample] = []
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(EvalSample(**payload))
    return samples


def save_evalset(samples: list[EvalSample], path: str) -> None:
    eval_path = Path(path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(sample), ensure_ascii=False) for sample in samples]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    eval_path.write_text(content, encoding="utf-8")
