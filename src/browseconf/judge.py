"""Judge adapters.

The DeepResearch contract follows Alibaba-NLP/DeepResearch at commit
f72f75d8c3eb842f2bbbab096a12206ff66e270f (Apache-2.0), with explicit failures.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .models import ChatModel
from .parsing import parse_json_object, parse_judge_correct
from .prompts import (
    ANSWER_EQUIVALENCE_PROMPT,
    BROWSECOMP_JUDGE_PROMPT,
    BROWSECOMP_JUDGE_RESPONSE_FORMAT,
)
from .schemas import Judgement


@dataclass(slots=True)
class DeepResearchJudgeAdapter:
    """Thin adapter for DeepResearch's official BrowseComp grading branch."""

    model: ChatModel
    temperature: float = 1.0
    max_tokens: int | None = None
    attempts: int = 5
    retry_delay_seconds: float = 3.0

    def grade(
        self,
        *,
        question_id: str,
        question: str,
        correct_answer: str,
        response: str,
    ) -> Judgement:
        prompt = BROWSECOMP_JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        last_error = "judge_api_failure"
        for attempt in range(self.attempts):
            try:
                result = self.model.generate(
                    [{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=BROWSECOMP_JUDGE_RESPONSE_FORMAT,
                )
                try:
                    payload = parse_json_object(result.content)
                    value = str(payload["correct"]).lower()
                    correct = value == "yes" if value in {"yes", "no"} else None
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    correct = parse_judge_correct(result.content)
                if correct is None and attempt + 1 < self.attempts:
                    if self.retry_delay_seconds:
                        time.sleep(self.retry_delay_seconds)
                    continue
                return Judgement(
                    question_id=question_id,
                    correct=correct,
                    raw_response=result.content,
                    usage=result.usage,
                    error=None if correct is not None else "judge_parse_failure",
                )
            except Exception as exc:  # noqa: BLE001 - provider errors are intentionally retried
                last_error = str(exc)
                if attempt + 1 < self.attempts and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
        return Judgement(
            question_id=question_id,
            correct=None,
            raw_response="",
            error=last_error,
        )


# Backward-compatible public name. It now uses the audited DeepResearch adapter.
BrowseCompJudge = DeepResearchJudgeAdapter


@dataclass(slots=True)
class SemanticEquivalenceJudge:
    model: ChatModel
    question: str
    temperature: float = 0.0

    def __call__(self, answer_a: str, answer_b: str) -> bool:
        prompt = ANSWER_EQUIVALENCE_PROMPT.format(
            question=self.question, answer_a=answer_a, answer_b=answer_b
        )
        result = self.model.generate(
            [{"role": "user", "content": prompt}], temperature=self.temperature, max_tokens=16
        )
        parsed = parse_judge_correct(result.content)
        if parsed is None:
            raise ValueError(f"Could not parse equivalence judgement: {result.content!r}")
        return parsed
