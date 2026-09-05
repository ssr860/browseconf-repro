from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(item) for item in value]
    return value


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens


@dataclass(slots=True)
class ModelResponse:
    content: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolEvent:
    index: int
    name: str
    arguments: dict[str, Any]
    result: str
    started_at: str
    duration_seconds: float
    ok: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolOutput:
    content: str
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Attempt:
    question_id: str
    attempt_index: int
    answer: str | None
    confidence: int
    raw_output: str
    messages: list[dict[str, str]]
    tool_events: list[ToolEvent] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    interaction_count: int = 0
    model_call_count: int = 0
    stop_reason: str = "unknown"
    error: str | None = None
    parse_warnings: list[str] = field(default_factory=list)
    started_at: str = ""
    duration_seconds: float = 0.0


@dataclass(slots=True)
class SummaryCall:
    after_attempt: int
    summary: str
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


@dataclass(slots=True)
class BrowseConfResult:
    question_id: str
    question: str
    method: Literal["zero", "summary", "neg"]
    threshold: int
    max_attempts: int
    attempts: list[Attempt]
    selected_attempt_index: int
    final_answer: str | None
    final_confidence: int
    stop_reason: str
    summaries: list[SummaryCall] = field(default_factory=list)
    model: str = ""
    config_hash: str = ""

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for attempt in self.attempts:
            total.add(attempt.usage)
        for call in self.summaries:
            total.add(call.usage)
        return total


@dataclass(slots=True)
class Question:
    question_id: str
    question: str
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Judgement:
    question_id: str
    correct: bool | None
    raw_response: str
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
