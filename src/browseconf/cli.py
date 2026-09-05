from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agent import TextReactAgent
from .aggregation import CandidateAnswer, cisc, self_consistency
from .calibration import select_threshold
from .datasets import BROWSECOMP_URL, limit_questions, load_browsecomp_csv, load_jsonl
from .manifest import write_manifest
from .metrics import result_report
from .models import ScriptedModel
from .runner import build_agent, build_judge, load_config, run_adaptive, run_fixed_pool
from .schemas import to_dict
from .storage import JsonlStore, atomic_write_json, read_jsonl
from .tools import StaticTool, ToolRouter


def _questions(args: argparse.Namespace):
    if args.dataset_format == "browsecomp":
        questions = load_browsecomp_csv(args.dataset or BROWSECOMP_URL)
    else:
        if not args.dataset:
            raise ValueError("--dataset is required for JSONL input")
        questions = load_jsonl(args.dataset)
    return limit_questions(questions, args.limit)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    if getattr(args, "backend", None):
        config["backend"] = args.backend
    return config


def _require_paid_authorization(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if config.get("backend") != "offline-mock" and not args.allow_paid_apis:
        raise RuntimeError(
            "Refusing a network-capable run without --allow-paid-apis. "
            "Review the config, credentials, limit and worker count first."
        )


def command_run(args: argparse.Namespace) -> None:
    config = _config(args)
    _require_paid_authorization(args, config)
    agent, summary_model = build_agent(config)
    pending, skipped = run_adaptive(
        _questions(args),
        agent=agent,
        summary_model=summary_model,
        method=args.method,
        threshold=args.threshold,
        max_attempts=args.max_attempts,
        output_path=args.output,
        workers=args.workers,
    )
    _manifest(args, config=config, inputs=[args.dataset] if args.dataset else [])
    print(json.dumps({"completed_now": pending, "skipped_existing": skipped}, indent=2))


def command_fixed_pool(args: argparse.Namespace) -> None:
    config = _config(args)
    _require_paid_authorization(args, config)
    agent, _ = build_agent(config)
    pending, skipped = run_fixed_pool(
        _questions(args),
        agent=agent,
        rollouts=args.rollouts,
        output_path=args.output,
        workers=args.workers,
    )
    _manifest(args, config=config, inputs=[args.dataset] if args.dataset else [])
    print(json.dumps({"completed_now": pending, "skipped_existing": skipped}, indent=2))


def command_aggregate(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["question_id"])].append(row)
    store = JsonlStore(Path(args.output), key_field="question_id")
    existing = store.keys()
    for question_id, group in sorted(groups.items()):
        if question_id in existing:
            continue
        group.sort(key=lambda row: int(row["rollout_index"]))
        candidates = [
            CandidateAnswer(
                answer=str(row.get("answer") or ""),
                confidence=int(row.get("confidence", -1)),
                index=int(row["rollout_index"]),
            )
            for row in group
            if row.get("answer")
        ]
        if not candidates:
            selected_answer = None
            selected_confidence = -1
            members: list[int] = []
            score = 0.0
        elif args.strategy == "pass1":
            selected = min(candidates, key=lambda candidate: candidate.index)
            selected_answer = selected.answer
            selected_confidence = selected.confidence
            members = [selected.index]
            score = 1.0
        else:
            aggregate = (
                self_consistency(candidates)
                if args.strategy == "sc"
                else cisc(candidates, temperature=args.temperature)
            )
            selected_answer = aggregate.answer
            members = aggregate.member_indices
            score = aggregate.score
            selected_confidence = max(
                candidate.confidence for candidate in candidates if candidate.index in members
            )
        store.append(
            {
                "question_id": question_id,
                "question": group[0].get("question", ""),
                "strategy": args.strategy,
                "final_answer": selected_answer,
                "final_confidence": selected_confidence,
                "member_indices": members,
                "score": score,
                "attempt_count": len(group),
                "stop_reason": "fixed_pool_aggregation",
            }
        )
    _manifest(args, inputs=[args.input])


def _prediction_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("question_id"))


def _prediction_text(row: dict[str, Any]) -> str:
    answer = row.get("final_answer", row.get("answer"))
    confidence = row.get("final_confidence", row.get("confidence", -1))
    return f"Answer: {answer or ''}\nConfidence: {confidence}"


def command_judge(args: argparse.Namespace) -> None:
    config = _config(args)
    _require_paid_authorization(args, config)
    judge = build_judge(config)
    question_map = {question.question_id: question for question in _questions(args)}
    store = JsonlStore(Path(args.output), key_field="prediction_id")
    existing = store.keys()
    rows = [row for row in read_jsonl(args.predictions) if _prediction_id(row) not in existing]

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        prediction_id = _prediction_id(row)
        question_id = str(row["question_id"])
        question = question_map.get(question_id)
        if question is None or question.answer is None:
            raise ValueError(f"No reference answer found for {question_id}")
        judgement = judge.grade(
            question_id=question_id,
            question=question.question,
            correct_answer=question.answer,
            response=_prediction_text(row),
        )
        payload = to_dict(judgement)
        payload["prediction_id"] = prediction_id
        return payload

    if args.workers == 1:
        for row in rows:
            store.append(execute(row))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute, row) for row in rows]
            for future in as_completed(futures):
                store.append(future.result())
    inputs = [args.predictions]
    if args.dataset:
        inputs.append(args.dataset)
    _manifest(args, config=config, inputs=inputs)


def command_calibrate(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.predictions)
    judgements = {
        str(row.get("prediction_id") or row["question_id"]): row.get("correct")
        for row in read_jsonl(args.judgements)
    }
    observations: list[tuple[int, bool]] = []
    for row in predictions:
        key = _prediction_id(row)
        correct = judgements.get(key)
        confidence = int(row.get("final_confidence", row.get("confidence", -1)))
        if correct is not None and 0 <= confidence <= 100:
            observations.append((confidence, bool(correct)))
    selection = select_threshold(observations, k=args.k, min_count=args.min_count)
    atomic_write_json(args.output, selection)
    _manifest(args, inputs=[args.predictions, args.judgements])
    print(json.dumps({"threshold": selection.threshold, "n": len(observations)}, indent=2))


def command_report(args: argparse.Namespace) -> None:
    report = result_report(read_jsonl(args.predictions), read_jsonl(args.judgements))
    atomic_write_json(args.output, report)
    _manifest(args, inputs=[args.predictions, args.judgements])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_fixed_report(args: argparse.Namespace) -> None:
    pool = read_jsonl(args.pool)
    judgements = {
        str(row.get("prediction_id") or row["question_id"]): row.get("correct")
        for row in read_jsonl(args.judgements)
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[str(row["question_id"])].append(row)
    pass1_values: list[bool] = []
    passk_values: list[bool] = []
    for group in groups.values():
        group.sort(key=lambda row: int(row["rollout_index"]))
        values = [bool(judgements.get(str(row["task_id"]), False)) for row in group[: args.k]]
        if values:
            pass1_values.append(values[0])
            passk_values.append(any(values))
    report = {
        "n_questions": len(pass1_values),
        "k": args.k,
        "pass_at_1": sum(pass1_values) / len(pass1_values) if pass1_values else None,
        "pass_at_k": sum(passk_values) / len(passk_values) if passk_values else None,
    }
    atomic_write_json(args.output, report)
    _manifest(args, inputs=[args.pool, args.judgements])
    print(json.dumps(report, indent=2))


def command_smoke(_: argparse.Namespace) -> None:
    model = ScriptedModel(
        [
            '<think>search</think><tool_call>{"name":"search","arguments":{"query":["x"]}}</tool_call>',
            "<answer>Answer: smoke-ok\nConfidence: 97</answer>",
        ]
    )
    search = StaticTool("search", "one result")
    visit = StaticTool("visit", "one page")
    agent = TextReactAgent(model=model, tools=ToolRouter.from_tools(search, visit))
    attempt = agent.run(question_id="smoke", question="test", attempt_index=1)
    if attempt.answer != "smoke-ok" or attempt.confidence != 97 or attempt.interaction_count != 1:
        raise RuntimeError(f"Smoke test failed: {attempt}")
    print(json.dumps({"status": "ok", "answer": attempt.answer, "confidence": 97}))


def command_demo(_: argparse.Namespace) -> None:
    model = ScriptedModel(
        [
            "<answer>Answer: first hypothesis\nConfidence: 35</answer>",
            "<answer>Answer: verified answer\nConfidence: 96</answer>",
        ]
    )
    agent = TextReactAgent(
        model=model,
        tools=ToolRouter.from_tools(
            StaticTool("search", "offline"), StaticTool("visit", "offline")
        ),
    )
    from .policies import BrowseConfRunner

    result = BrowseConfRunner(agent, method="neg", threshold=90, max_attempts=3).run(
        "demo-1", "Which answer is supported?"
    )
    selection = select_threshold([(35, False), (70, False), (91, True), (96, True)], k=1)
    aggregate = cisc(
        [
            CandidateAnswer("first hypothesis", 35, 1),
            CandidateAnswer("verified answer", 96, 2),
        ],
        temperature=10,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "adaptive": {
                    "answer": result.final_answer,
                    "attempts": result.attempt_count,
                    "stop_reason": result.stop_reason,
                },
                "calibrated_threshold": selection.threshold,
                "cisc_answer": aggregate.answer,
            },
            indent=2,
        )
    )


def _manifest(
    args: argparse.Namespace,
    *,
    config: dict[str, Any] | None = None,
    inputs: list[str | Path] | None = None,
) -> None:
    arguments = {
        key: value
        for key, value in vars(args).items()
        if key not in {"func"} and isinstance(value, (str, int, float, bool, type(None)))
    }
    write_manifest(
        args.output,
        command=str(args.command),
        arguments=arguments,
        config=config,
        inputs=inputs,
    )


def _dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", help="JSONL/CSV path; optional for official BrowseComp URL")
    parser.add_argument("--dataset-format", choices=["jsonl", "browsecomp"], default="jsonl")
    parser.add_argument("--limit", type=int)


def _backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["paper-browseconf", "official-deepresearch", "local", "offline-mock"],
        help="Override the backend selected in the JSON config",
    )
    parser.add_argument(
        "--allow-paid-apis",
        action="store_true",
        help="Explicitly authorize network-capable model/search/Jina calls",
    )


def command_doctor(args: argparse.Namespace) -> None:
    """Validate a configuration without making network or paid API calls."""
    config = _config(args)
    backend = config.get("backend", "official-deepresearch")
    checks: list[dict[str, Any]] = []

    def credential(role: str, section: str, default_env: str) -> None:
        env_name = str(config.get(section, {}).get("api_key_env", default_env))
        checks.append(
            {"role": role, "environment_variable": env_name, "present": bool(os.getenv(env_name))}
        )

    if backend != "offline-mock":
        credential("agent_model", "agent_model", "OPENAI_API_KEY")
        credential("summary_model", "summary_model", "OPENAI_API_KEY")
        credential("judge_model", "judge_model", "OPENAI_API_KEY")
        credential("serper_search", "search", "SERPER_KEY_ID")
        credential("jina_reader_optional", "visit", "JINA_API_KEYS")

    warnings: list[str] = []
    if backend == "paper-browseconf":
        agent_cfg = config.get("agent", {})
        expected = {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_context_tokens": 131_072,
        }
        for name, value in expected.items():
            if agent_cfg.get(name) != value:
                warnings.append(f"paper mismatch: agent.{name} must be {value!r}")
        if config.get("summary_model", {}).get("model_id") != "gpt-oss-120b":
            warnings.append("paper mismatch: summary_model.model_id must be gpt-oss-120b")
        if config.get("judge_model", {}).get("model_id") != "gpt-4o-2024-08-06":
            warnings.append("benchmark mismatch: judge_model.model_id must be gpt-4o-2024-08-06")
    placeholder_sections = []
    for section in ("agent_model", "summary_model", "judge_model"):
        value = str(config.get(section, {}).get("base_url", ""))
        if "replace" in value.lower() or "example" in value.lower():
            placeholder_sections.append(section)
    if placeholder_sections:
        warnings.append("replace placeholder base_url in: " + ", ".join(placeholder_sections))
    payload = {
        "backend": backend,
        "network_calls_made": False,
        "credentials": checks,
        "missing_required": [
            item["environment_variable"]
            for item in checks
            if not item["present"] and item["role"] != "jina_reader_optional"
        ],
        "warnings": warnings,
        "ready": not warnings
        and all(item["present"] or item["role"] == "jina_reader_optional" for item in checks),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="browseconf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a BrowseConf adaptive policy")
    run.add_argument("--config", required=True)
    _backend_argument(run)
    _dataset_arguments(run)
    run.add_argument("--method", choices=["zero", "summary", "neg"], required=True)
    run.add_argument("--threshold", type=int, required=True)
    run.add_argument("--max-attempts", type=int, default=10)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--output", required=True)
    run.set_defaults(func=command_run)

    fixed = subparsers.add_parser("fixed-pool", help="Generate a reusable rollout pool")
    fixed.add_argument("--config", required=True)
    _backend_argument(fixed)
    _dataset_arguments(fixed)
    fixed.add_argument("--rollouts", type=int, default=10)
    fixed.add_argument("--workers", type=int, default=1)
    fixed.add_argument("--output", required=True)
    fixed.set_defaults(func=command_fixed_pool)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate a fixed rollout pool")
    aggregate.add_argument("--input", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--strategy", choices=["pass1", "sc", "cisc"], required=True)
    aggregate.add_argument("--temperature", type=float, default=10.0)
    aggregate.set_defaults(func=command_aggregate)

    judge = subparsers.add_parser("judge", help="Judge predictions against references")
    judge.add_argument("--config", required=True)
    _backend_argument(judge)
    _dataset_arguments(judge)
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--workers", type=int, default=100)
    judge.add_argument("--output", required=True)
    judge.set_defaults(func=command_judge)

    calibrate = subparsers.add_parser("calibrate", help="Select confidence threshold")
    calibrate.add_argument("--predictions", required=True)
    calibrate.add_argument("--judgements", required=True)
    calibrate.add_argument("--k", type=float, default=10)
    calibrate.add_argument("--min-count", type=int, default=1)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(func=command_calibrate)

    report = subparsers.add_parser("report", help="Create accuracy/confidence report")
    report.add_argument("--predictions", required=True)
    report.add_argument("--judgements", required=True)
    report.add_argument("--output", required=True)
    report.set_defaults(func=command_report)

    fixed_report = subparsers.add_parser("fixed-report", help="Calculate Pass@1 and Pass@K")
    fixed_report.add_argument("--pool", required=True)
    fixed_report.add_argument("--judgements", required=True)
    fixed_report.add_argument("--k", type=int, default=10)
    fixed_report.add_argument("--output", required=True)
    fixed_report.set_defaults(func=command_fixed_report)

    smoke = subparsers.add_parser("smoke", help="Run a zero-cost internal smoke test")
    smoke.set_defaults(func=command_smoke)

    demo = subparsers.add_parser("demo", help="Run an offline adaptive/aggregation demo")
    demo.set_defaults(func=command_demo)

    doctor = subparsers.add_parser("doctor", help="Validate config and credentials without API calls")
    doctor.add_argument("--config", required=True)
    _backend_argument(doctor)
    doctor.set_defaults(func=command_doctor)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
