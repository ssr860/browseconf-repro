from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agent import TextReactAgent
from .judge import DeepResearchJudgeAdapter
from .models import ChatModel, ScriptedModel, model_from_config
from .policies import BrowseConfRunner, TrajectorySummarizer
from .schemas import Question, to_dict
from .storage import JsonlStore
from .tools import (
    DeepResearchJinaReader,
    DeepResearchSearchAdapter,
    DeepResearchVisitAdapter,
    FileCache,
    JinaReader,
    SerperSearchTool,
    StaticTool,
    ToolRouter,
    VisitTool,
)


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_agent(config: dict[str, Any]) -> tuple[TextReactAgent, ChatModel]:
    backend = config.get("backend", "official-deepresearch")
    if backend == "offline-mock":
        offline = config.get("offline", {})
        agent_model = ScriptedModel(list(offline.get("agent_responses", [])))
        summary_model = ScriptedModel(list(offline.get("summary_responses", [])))
        agent = TextReactAgent(
            model=agent_model,
            tools=ToolRouter.from_tools(
                StaticTool("search", str(offline.get("search_result", "offline search"))),
                StaticTool("visit", str(offline.get("visit_result", "offline page"))),
            ),
            protocol=str(offline.get("protocol", "official-deepresearch")),
        )
        return agent, summary_model
    if backend not in {"paper-browseconf", "official-deepresearch", "local"}:
        raise ValueError(f"Unknown backend: {backend}")
    agent_model = model_from_config(config["agent_model"])
    summary_model = model_from_config(config.get("summary_model", config["agent_model"]))
    cache = FileCache(Path(config.get("cache_dir", "artifacts/cache")))
    search_cfg = config.get("search", {})
    visit_cfg = config.get("visit", {})
    if backend in {"paper-browseconf", "official-deepresearch"}:
        search = DeepResearchSearchAdapter(
            api_key_env=search_cfg.get("api_key_env", "SERPER_KEY_ID"),
            top_k=int(search_cfg.get("top_k", 10)),
            timeout_seconds=float(search_cfg.get("timeout_seconds", 30)),
            retries=int(search_cfg.get("retries", 5)),
            cache=cache,
        )
        reader = DeepResearchJinaReader(
            api_key_env=visit_cfg.get("api_key_env", "JINA_API_KEYS"),
            timeout_seconds=float(visit_cfg.get("timeout_seconds", 50)),
            request_retries=int(visit_cfg.get("request_retries", 3)),
            outer_retries=int(visit_cfg.get("outer_retries", 8)),
            cache=cache,
        )
        visit = DeepResearchVisitAdapter(
            reader=reader,
            summary_model=summary_model,
            summary_temperature=float(visit_cfg.get("summary_temperature", 0.7)),
            summary_max_tokens=visit_cfg.get("summary_max_tokens"),
            max_page_tokens=int(visit_cfg.get("max_page_tokens", 95_000)),
            summary_retries=int(visit_cfg.get("summary_retries", 3)),
            parse_retries=int(visit_cfg.get("parse_retries", 3)),
            batch_timeout_seconds=float(visit_cfg.get("batch_timeout_seconds", 900)),
        )
    else:
        search = SerperSearchTool(
            api_key_env=search_cfg.get("api_key_env", "GOOGLE_SEARCH_KEY"),
            country=search_cfg.get("country", "us"),
            language=search_cfg.get("language"),
            top_k=int(search_cfg.get("top_k", 10)),
            workers=int(search_cfg.get("workers", 3)),
            timeout_seconds=float(search_cfg.get("timeout_seconds", 30)),
            cache=cache,
        )
        reader = JinaReader(
            api_key_env=visit_cfg.get("api_key_env", "JINA_API_KEY"),
            max_chars=int(visit_cfg.get("max_chars", 150_000)),
            timeout_seconds=float(visit_cfg.get("timeout_seconds", 30)),
            cache=cache,
        )
        visit = VisitTool(
            reader=reader,
            summary_model=summary_model,
            workers=int(visit_cfg.get("workers", 3)),
            summary_temperature=float(visit_cfg.get("summary_temperature", 0.0)),
            summary_max_tokens=int(visit_cfg.get("summary_max_tokens", 4096)),
        )
    agent_cfg = config.get("agent", {})
    agent = TextReactAgent(
        model=agent_model,
        tools=ToolRouter.from_tools(search, visit),
        max_interactions=int(
            agent_cfg.get(
                "max_interactions",
                100 if backend in {"paper-browseconf", "official-deepresearch"} else 80,
            )
        ),
        max_context_tokens=int(
            agent_cfg.get(
                "max_context_tokens",
                131_072
                if backend == "paper-browseconf"
                else (110 * 1024 if backend == "official-deepresearch" else 131_072),
            )
        ),
        max_output_tokens=agent_cfg.get(
            "max_output_tokens",
            10_000 if backend in {"paper-browseconf", "official-deepresearch"} else None,
        ),
        temperature=float(agent_cfg.get("temperature", 0.6)),
        top_p=float(agent_cfg.get("top_p", 0.95)),
        presence_penalty=agent_cfg.get(
            "presence_penalty", 1.1 if backend == "official-deepresearch" else None
        ),
        logprobs=agent_cfg.get("logprobs", True if backend == "official-deepresearch" else None),
        protocol=backend,
        max_duration_seconds=(
            float(agent_cfg["max_duration_seconds"])
            if "max_duration_seconds" in agent_cfg
            else (150 * 60 if backend in {"paper-browseconf", "official-deepresearch"} else None)
        ),
    )
    return agent, summary_model


def build_judge(config: dict[str, Any]) -> DeepResearchJudgeAdapter:
    if config.get("backend") == "offline-mock":
        return DeepResearchJudgeAdapter(
            model=ScriptedModel(list(config.get("offline", {}).get("judge_responses", []))),
            attempts=1,
            retry_delay_seconds=0,
        )
    judge_cfg = config["judge_model"]
    return DeepResearchJudgeAdapter(
        model=model_from_config(judge_cfg),
        temperature=float(judge_cfg.get("temperature", 1.0)),
        max_tokens=judge_cfg.get("max_tokens"),
        attempts=int(judge_cfg.get("judge_attempts", 5)),
        retry_delay_seconds=float(judge_cfg.get("retry_delay_seconds", 3.0)),
    )


def run_adaptive(
    questions: Iterable[Question],
    *,
    agent: TextReactAgent,
    summary_model: ChatModel,
    method: str,
    threshold: int,
    max_attempts: int,
    output_path: str | Path,
    workers: int = 1,
) -> tuple[int, int]:
    store = JsonlStore(Path(output_path), key_field="question_id")
    completed = store.keys()
    pending = [question for question in questions if question.question_id not in completed]
    runner = BrowseConfRunner(
        attempt_runner=agent,
        method=method,  # type: ignore[arg-type]
        threshold=threshold,
        max_attempts=max_attempts,
        summarizer=TrajectorySummarizer(summary_model) if method == "summary" else None,
    )

    def execute(question: Question) -> dict[str, Any]:
        result = runner.run(question.question_id, question.question)
        payload = to_dict(result)
        payload["attempt_count"] = result.attempt_count
        payload["usage"] = to_dict(result.usage)
        return payload

    if workers == 1:
        for question in pending:
            store.append(execute(question))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute, question): question for question in pending}
            for future in as_completed(futures):
                store.append(future.result())
    return len(pending), len(completed)


def run_fixed_pool(
    questions: Iterable[Question],
    *,
    agent: TextReactAgent,
    rollouts: int,
    output_path: str | Path,
    workers: int = 1,
) -> tuple[int, int]:
    store = JsonlStore(Path(output_path), key_field="task_id")
    completed = store.keys()
    tasks = [
        (question, rollout)
        for question in questions
        for rollout in range(1, rollouts + 1)
        if f"{question.question_id}::{rollout}" not in completed
    ]

    def execute(task: tuple[Question, int]) -> dict[str, Any]:
        question, rollout = task
        attempt = agent.run(
            question_id=question.question_id,
            question=question.question,
            attempt_index=rollout,
        )
        return {
            "task_id": f"{question.question_id}::{rollout}",
            "question_id": question.question_id,
            "question": question.question,
            "rollout_index": rollout,
            "answer": attempt.answer,
            "confidence": attempt.confidence,
            "stop_reason": attempt.stop_reason,
            "attempt": to_dict(attempt),
        }

    if workers == 1:
        for task in tasks:
            store.append(execute(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                store.append(future.result())
    return len(tasks), len(completed)
