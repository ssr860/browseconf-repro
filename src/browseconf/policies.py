from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from .models import ChatModel
from .prompts import (
    INITIAL_SUMMARY_PROMPT,
    NEG_QUESTION,
    SUBSEQUENT_SUMMARY_PROMPT,
    SUMMARY_QUESTION,
)
from .schemas import Attempt, BrowseConfResult, SummaryCall


class AttemptRunner(Protocol):
    model: ChatModel

    def run(self, *, question_id: str, question: str, attempt_index: int) -> Attempt: ...


@dataclass(slots=True)
class TrajectorySummarizer:
    model: ChatModel
    temperature: float = 0.0
    max_tokens: int = 8192

    def summarize(
        self,
        question: str,
        attempt: Attempt,
        previous_summary: str | None,
    ) -> SummaryCall:
        searches = "\n\n".join(
            event.result for event in attempt.tool_events if event.name == "search"
        )
        pages = "\n\n".join(event.result for event in attempt.tool_events if event.name == "visit")
        if previous_summary is None:
            prompt = INITIAL_SUMMARY_PROMPT.format(
                question=question, search_results=searches, webpage_contents=pages
            )
        else:
            prompt = SUBSEQUENT_SUMMARY_PROMPT.format(
                question=question,
                previous_summary=previous_summary,
                search_results=searches,
                webpage_contents=pages,
            )
        try:
            response = self.model.generate(
                [{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return SummaryCall(
                after_attempt=attempt.attempt_index,
                summary=response.content.strip(),
                usage=response.usage,
            )
        except Exception as exc:  # noqa: BLE001 - model adapters can raise arbitrary errors
            return SummaryCall(
                after_attempt=attempt.attempt_index,
                summary=previous_summary or "",
                error=str(exc),
            )


@dataclass(slots=True)
class BrowseConfRunner:
    attempt_runner: AttemptRunner
    method: Literal["zero", "summary", "neg"] = "zero"
    threshold: int = 95
    max_attempts: int = 10
    summarizer: TrajectorySummarizer | None = None

    def __post_init__(self) -> None:
        if self.method not in {"zero", "summary", "neg"}:
            raise ValueError(f"Unknown BrowseConf method: {self.method}")
        if not 0 <= self.threshold <= 100:
            raise ValueError("threshold must be in [0, 100]")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.method == "summary" and self.summarizer is None:
            raise ValueError("summary method requires a TrajectorySummarizer")

    def run(self, question_id: str, question: str) -> BrowseConfResult:
        attempts: list[Attempt] = []
        summaries: list[SummaryCall] = []
        previous_summary: str | None = None
        negative_answers: list[str] = []

        for index in range(1, self.max_attempts + 1):
            conditioned_question = question
            if self.method == "summary" and previous_summary:
                conditioned_question = SUMMARY_QUESTION.format(
                    question=question, summary=previous_summary
                )
            elif self.method == "neg" and negative_answers:
                conditioned_question = NEG_QUESTION.format(
                    question=question,
                    answers="\n".join(f"- {answer}" for answer in negative_answers),
                )
            attempt = self.attempt_runner.run(
                question_id=question_id,
                question=conditioned_question,
                attempt_index=index,
            )
            attempts.append(attempt)
            if attempt.confidence >= self.threshold:
                return self._result(question_id, question, attempts, summaries, index, "threshold")

            if index >= self.max_attempts:
                continue
            if self.method == "neg" and attempt.answer:
                negative_answers.append(attempt.answer)
            elif self.method == "summary":
                assert self.summarizer is not None
                summary_call = self.summarizer.summarize(question, attempt, previous_summary)
                summaries.append(summary_call)
                if summary_call.summary:
                    previous_summary = summary_call.summary

        selected = max(attempts, key=lambda attempt: attempt.confidence)
        return self._result(
            question_id,
            question,
            attempts,
            summaries,
            selected.attempt_index,
            "budget_exhausted",
        )

    def _result(
        self,
        question_id: str,
        question: str,
        attempts: list[Attempt],
        summaries: list[SummaryCall],
        selected_index: int,
        stop_reason: str,
    ) -> BrowseConfResult:
        selected = next(attempt for attempt in attempts if attempt.attempt_index == selected_index)
        config = {
            "method": self.method,
            "threshold": self.threshold,
            "max_attempts": self.max_attempts,
            "model": self.attempt_runner.model.model_id,
        }
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        return BrowseConfResult(
            question_id=question_id,
            question=question,
            method=self.method,
            threshold=self.threshold,
            max_attempts=self.max_attempts,
            attempts=attempts,
            selected_attempt_index=selected_index,
            final_answer=selected.answer,
            final_confidence=selected.confidence,
            stop_reason=stop_reason,
            summaries=summaries,
            model=self.attempt_runner.model.model_id,
            config_hash=config_hash,
        )
