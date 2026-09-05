from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from .schemas import Question

BROWSECOMP_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def _read_text(path_or_url: str | Path) -> str:
    value = str(path_or_url)
    if value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=60) as response:
            return response.read().decode("utf-8")
    return Path(value).read_text(encoding="utf-8-sig")


def load_browsecomp_csv(path_or_url: str | Path = BROWSECOMP_URL) -> list[Question]:
    rows = csv.DictReader(io.StringIO(_read_text(path_or_url)))
    questions: list[Question] = []
    for index, row in enumerate(rows):
        canary = row.get("canary", "")
        problem = decrypt(row.get("problem", ""), canary)
        answer = decrypt(row.get("answer", ""), canary)
        questions.append(
            Question(
                question_id=row.get("id") or f"browsecomp-{index:04d}",
                question=problem,
                answer=answer,
                metadata={"dataset": "BrowseComp", "row_index": index},
            )
        )
    return questions


def load_jsonl(path: str | Path) -> list[Question]:
    questions: list[Question] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question at line {index + 1}")
            questions.append(
                Question(
                    question_id=str(row.get("question_id") or row.get("id") or f"q-{index:06d}"),
                    question=question.strip(),
                    answer=row.get("answer"),
                    metadata=dict(row.get("metadata", {})),
                )
            )
    return questions


def limit_questions(questions: Iterable[Question], limit: int | None) -> list[Question]:
    rows = list(questions)
    return rows if limit is None else rows[:limit]
