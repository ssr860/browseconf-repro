from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schemas import ModelResponse, Usage


class ChatModel(Protocol):
    model_id: str

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse: ...


@dataclass(slots=True)
class OpenAICompatibleModel:
    """Minimal dependency-free client for an OpenAI-compatible chat endpoint."""

    model_id: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    default_temperature: float = 0.6
    default_top_p: float = 0.95
    default_max_tokens: int | None = None
    timeout_seconds: float = 300.0
    max_retries: int = 5
    extra_body: dict[str, Any] = field(default_factory=dict)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        presence_penalty: float | None = None,
        logprobs: bool | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "temperature": self.default_temperature if temperature is None else temperature,
            "top_p": self.default_top_p if top_p is None else top_p,
        }
        resolved_max = self.default_max_tokens if max_tokens is None else max_tokens
        if resolved_max is not None:
            payload["max_tokens"] = resolved_max
        if stop:
            payload["stop"] = list(stop)
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if response_format is not None:
            payload["response_format"] = response_format
        payload.update(self.extra_body)

        api_key = os.getenv(self.api_key_env, "EMPTY")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return _parse_openai_response(raw)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                last_error = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 401, 403, 404}:
                    break
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Model request failed after {self.max_retries} attempts: {last_error}")


def _parse_openai_response(raw: dict[str, Any]) -> ModelResponse:
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError("Model response contains no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        content = ""
    usage_raw = raw.get("usage") or {}
    completion_details = usage_raw.get("completion_tokens_details") or {}
    prompt_details = usage_raw.get("prompt_tokens_details") or {}
    usage = Usage(
        prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
        completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
        cached_tokens=int(prompt_details.get("cached_tokens") or 0),
    )
    return ModelResponse(
        content=content,
        usage=usage,
        model=str(raw.get("model") or ""),
        finish_reason=choice.get("finish_reason"),
        raw=raw,
    )


@dataclass(slots=True)
class ScriptedModel:
    """Deterministic model used by tests and zero-cost smoke runs."""

    responses: list[str]
    model_id: str = "scripted-model"
    calls: list[list[dict[str, str]]] = field(default_factory=list)
    call_options: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        **options: Any,
    ) -> ModelResponse:
        self.calls.append([dict(message) for message in messages])
        self.call_options.append(dict(options))
        if not self.responses:
            raise RuntimeError("ScriptedModel has no responses left")
        return ModelResponse(content=self.responses.pop(0), model=self.model_id)


def model_from_config(config: dict[str, Any]) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        model_id=config["model_id"],
        base_url=config.get("base_url", "http://127.0.0.1:8000/v1"),
        api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
        default_temperature=float(config.get("temperature", 0.6)),
        default_top_p=float(config.get("top_p", 0.95)),
        default_max_tokens=config.get("max_tokens"),
        timeout_seconds=float(config.get("timeout_seconds", 300)),
        max_retries=int(config.get("max_retries", 5)),
        extra_body=dict(config.get("extra_body", {})),
    )
