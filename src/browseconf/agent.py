"""Agent loop.

Portions adapted from Alibaba-NLP/DeepResearch@f72f75d8c3eb842f2bbbab096a12206ff66e270f
(Apache-2.0), modified for dependency injection and BrowseConf confidence policies.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass

from .models import ChatModel
from .parsing import parse_answer_and_confidence, parse_tool_call
from .prompts import (
    CONFIDENCE_SYSTEM_PROMPT,
    DEEPRESEARCH_SYSTEM_PROMPT,
    PAPER_TOOL_PROTOCOL,
    TOOLS_AND_PROTOCOL,
)
from .schemas import Attempt, ToolEvent, Usage
from .tools import ToolRouter


def approximate_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative tokenizer-free estimate; provider usage remains authoritative."""
    text = "\n".join(message.get("content", "") for message in messages)
    ascii_chars = sum(ord(char) < 128 for char in text)
    non_ascii_chars = len(text) - ascii_chars
    return (ascii_chars + 3) // 4 + non_ascii_chars


@dataclass(slots=True)
class TextReactAgent:
    model: ChatModel
    tools: ToolRouter
    system_prompt: str = CONFIDENCE_SYSTEM_PROMPT
    max_interactions: int = 80
    max_context_tokens: int = 131_072
    max_output_tokens: int | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    presence_penalty: float | None = None
    logprobs: bool | None = None
    protocol: str = "local"
    max_duration_seconds: float | None = None
    token_counter: Callable[[list[dict[str, str]]], int] = approximate_tokens

    def run(
        self,
        *,
        question_id: str,
        question: str,
        attempt_index: int,
    ) -> Attempt:
        started = time.monotonic()
        started_at = dt.datetime.now(dt.UTC).isoformat()
        if self.protocol == "official-deepresearch":
            system_prompt = DEEPRESEARCH_SYSTEM_PROMPT + dt.datetime.now(dt.UTC).date().isoformat()
            user_prompt = question
        elif self.protocol == "paper-browseconf":
            system_prompt = CONFIDENCE_SYSTEM_PROMPT + PAPER_TOOL_PROTOCOL
            user_prompt = question
        elif self.protocol == "local":
            system_prompt = self.system_prompt
            # The protocol contains literal JSON braces, so avoid str.format here.
            user_prompt = TOOLS_AND_PROTOCOL.replace("{question}", question)
        else:
            raise ValueError(f"Unknown agent protocol: {self.protocol}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tool_events: list[ToolEvent] = []
        total_usage = Usage()
        raw_output = ""
        error: str | None = None
        stop_reason = "interaction_limit"
        model_call_count = 0

        for interaction in range(1, self.max_interactions + 1):
            if (
                self.max_duration_seconds is not None
                and time.monotonic() - started > self.max_duration_seconds
            ):
                stop_reason = "time_limit"
                error = f"Attempt exceeded {self.max_duration_seconds:g} seconds"
                break
            if self.token_counter(messages) >= self.max_context_tokens:
                if self.protocol != "official-deepresearch":
                    stop_reason = "context_overflow"
                    error = "Estimated context length reached before model call"
                    break
                force_answer = (
                    "You have now reached the maximum context length you can handle. "
                    "You should stop making tool calls and, based on all the information above, "
                    "think again and provide what you consider the most likely answer in the "
                    "following format:<think>your final thinking</think>\n<answer>Answer: your "
                    "answer\nConfidence: an integer from 0 to 100</answer>"
                )
                if messages[-1]["role"] == "user":
                    messages[-1] = {"role": "user", "content": force_answer}
                else:
                    messages.append({"role": "user", "content": force_answer})
                try:
                    response = self.model.generate(
                        messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_output_tokens,
                        stop=["\n<tool_response>", "<tool_response>"],
                        presence_penalty=self.presence_penalty,
                        logprobs=self.logprobs,
                    )
                    total_usage.add(response.usage)
                    model_call_count += 1
                    raw_output = response.content.strip()
                    messages.append({"role": "assistant", "content": raw_output})
                    stop_reason = (
                        "answer_context_limit"
                        if "<answer>" in raw_output.lower()
                        else "context_overflow"
                    )
                    if stop_reason == "context_overflow":
                        error = "Forced answer at context limit had invalid format"
                except Exception as exc:  # noqa: BLE001
                    stop_reason = "context_overflow"
                    error = str(exc)
                break
            try:
                response = self.model.generate(
                    messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_output_tokens,
                    stop=["\n<tool_response>", "<tool_response>"],
                    presence_penalty=self.presence_penalty,
                    logprobs=self.logprobs,
                )
            except Exception as exc:  # noqa: BLE001 - external adapters can raise arbitrary errors
                message = str(exc)
                stop_reason = "context_overflow" if "context" in message.lower() else "model_error"
                error = message
                break
            total_usage.add(response.usage)
            model_call_count += 1
            raw_output = response.content.strip()
            messages.append({"role": "assistant", "content": raw_output})

            if "<answer>" in raw_output.lower() and "</answer>" in raw_output.lower():
                stop_reason = "answer"
                break

            try:
                tool_call = parse_tool_call(raw_output)
            except (TypeError, ValueError) as exc:
                tool_call = None
                tool_text = f"Error: {exc}"
                messages.append(
                    {"role": "user", "content": f"<tool_response>\n{tool_text}\n</tool_response>"}
                )
                continue

            if tool_call is None:
                parsed = parse_answer_and_confidence(raw_output)
                if parsed.answer is not None and parsed.confidence >= 0:
                    stop_reason = "answer_without_xml_tag"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your response contained neither a valid tool call nor a final answer. "
                            "Continue using the required XML format."
                        ),
                    }
                )
                continue

            tool_name, arguments = tool_call
            tool_started = time.monotonic()
            event_started_at = dt.datetime.now(dt.UTC).isoformat()
            try:
                tool_output = self.tools.call(tool_name, arguments)
                total_usage.add(tool_output.usage)
                tool_text = tool_output.content
                ok = True
                tool_error = None
                metadata = tool_output.metadata
            except Exception as exc:  # noqa: BLE001 - tool plugins can raise arbitrary errors
                tool_text = f"Error while calling {tool_name}: {exc}"
                ok = False
                tool_error = str(exc)
                metadata = {}
            tool_events.append(
                ToolEvent(
                    index=len(tool_events) + 1,
                    name=tool_name,
                    arguments=arguments,
                    result=tool_text,
                    started_at=event_started_at,
                    duration_seconds=time.monotonic() - tool_started,
                    ok=ok,
                    error=tool_error,
                    metadata=metadata,
                )
            )
            messages.append(
                {"role": "user", "content": f"<tool_response>\n{tool_text}\n</tool_response>"}
            )

        parsed = (
            parse_answer_and_confidence(raw_output) if stop_reason.startswith("answer") else None
        )
        answer = parsed.answer if parsed else None
        confidence = parsed.confidence if parsed else -1
        warnings = parsed.warnings if parsed else []
        return Attempt(
            question_id=question_id,
            attempt_index=attempt_index,
            answer=answer,
            confidence=confidence,
            raw_output=raw_output,
            messages=messages,
            tool_events=tool_events,
            usage=total_usage,
            interaction_count=len(tool_events),
            model_call_count=model_call_count,
            stop_reason=stop_reason,
            error=error,
            parse_warnings=warnings,
            started_at=started_at,
            duration_seconds=time.monotonic() - started,
        )
