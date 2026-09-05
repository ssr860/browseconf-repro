from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
ANSWER_FIELD_RE = re.compile(
    r"(?:\*\*)?Answer(?:\*\*)?\s*:\s*(.+?)(?=\n\s*(?:\*\*)?Confidence(?:\*\*)?\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
CONFIDENCE_RE = re.compile(
    r"(?:\*\*)?Confidence(?:\*\*)?\s*:\s*(-?\d{1,3})(?:\s*%)?",
    re.IGNORECASE,
)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class ParsedAnswer:
    answer: str | None
    confidence: int
    warnings: list[str] = field(default_factory=list)


def _answer_region(text: str) -> str:
    matches = ANSWER_TAG_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def parse_answer_and_confidence(text: str) -> ParsedAnswer:
    region = _answer_region(text)
    warnings: list[str] = []
    answer_match = ANSWER_FIELD_RE.search(region)
    if answer_match:
        answer = answer_match.group(1).strip()
    elif region and not TOOL_CALL_RE.search(region):
        answer = region.strip()
        warnings.append("answer_field_missing")
    else:
        answer = None
        warnings.append("answer_missing")

    confidence_matches = CONFIDENCE_RE.findall(region)
    if not confidence_matches:
        confidence = -1
        warnings.append("confidence_missing")
    else:
        confidence = int(confidence_matches[-1])
        if not 0 <= confidence <= 100:
            confidence = -1
            warnings.append("confidence_out_of_range")
        if len(confidence_matches) > 1:
            warnings.append("multiple_confidence_fields")

    if answer is not None:
        answer = answer.strip().strip("` ") or None
    return ParsedAnswer(answer=answer, confidence=confidence, warnings=warnings)


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        # DeepResearch uses json5.loads. Supporting Python/JSON5-style quoted
        # objects here covers its emitted single quotes and trailing commas
        # without adding a runtime dependency or accepting executable input.
        try:
            payload = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as fallback_exc:
            raise ValueError(f"Invalid tool-call JSON: {exc}") from fallback_exc
    if not isinstance(payload, dict):
        raise TypeError("Tool call must be a JSON object")
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ValueError("Tool call requires a non-empty string 'name'")
    if not isinstance(arguments, dict):
        raise TypeError("Tool call 'arguments' must be an object")
    return name, arguments


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right < left:
            raise ValueError("Response does not contain a JSON object")
        value = json.loads(text[left : right + 1])
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def parse_judge_correct(text: str) -> bool | None:
    matches = re.findall(r"correct\s*:\s*(yes|no)", text, re.IGNORECASE)
    if not matches:
        stripped = text.strip().lower()
        if stripped in {"yes", "correct"}:
            return True
        if stripped in {"no", "incorrect"}:
            return False
        return None
    return matches[-1].lower() == "yes"
