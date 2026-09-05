import json
from pathlib import Path

import pytest

from browseconf.cli import main


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_offline_demo(capsys):
    main(["demo"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["adaptive"]["answer"] == "verified answer"
    assert payload["adaptive"]["attempts"] == 2
    assert payload["cisc_answer"] == "verified answer"


def test_aggregate_calibrate_and_report_create_manifests(tmp_path):
    pool = tmp_path / "pool.jsonl"
    write_jsonl(
        pool,
        [
            {
                "task_id": "q1::1",
                "question_id": "q1",
                "question": "Q",
                "rollout_index": 1,
                "answer": "wrong",
                "confidence": 20,
            },
            {
                "task_id": "q1::2",
                "question_id": "q1",
                "question": "Q",
                "rollout_index": 2,
                "answer": "right",
                "confidence": 95,
            },
        ],
    )
    aggregated = tmp_path / "cisc.jsonl"
    main(
        [
            "aggregate",
            "--input",
            str(pool),
            "--output",
            str(aggregated),
            "--strategy",
            "cisc",
        ]
    )
    assert json.loads(aggregated.read_text(encoding="utf-8"))["final_answer"] == "right"
    assert (tmp_path / "cisc.jsonl.manifest.json").exists()

    judgements = tmp_path / "judgements.jsonl"
    write_jsonl(
        judgements,
        [
            {"prediction_id": "q1::1", "question_id": "q1", "correct": False},
            {"prediction_id": "q1::2", "question_id": "q1", "correct": True},
        ],
    )
    threshold = tmp_path / "threshold.json"
    main(
        [
            "calibrate",
            "--predictions",
            str(pool),
            "--judgements",
            str(judgements),
            "--k",
            "10",
            "--output",
            str(threshold),
        ]
    )
    assert json.loads(threshold.read_text(encoding="utf-8"))["threshold"] == 21
    assert (tmp_path / "threshold.json.manifest.json").exists()

    passk = tmp_path / "passk.json"
    main(
        [
            "fixed-report",
            "--pool",
            str(pool),
            "--judgements",
            str(judgements),
            "--k",
            "2",
            "--output",
            str(passk),
        ]
    )
    report = json.loads(passk.read_text(encoding="utf-8"))
    assert report["pass_at_1"] == 0.0
    assert report["pass_at_k"] == 1.0


def test_offline_mock_run_and_resume(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    write_jsonl(dataset, [{"question_id": "q1", "question": "Q", "answer": "offline answer"}])
    output = tmp_path / "predictions.jsonl"
    config = Path(__file__).parents[1] / "configs" / "mock.json"
    args = [
        "run",
        "--config",
        str(config),
        "--dataset",
        str(dataset),
        "--method",
        "zero",
        "--threshold",
        "95",
        "--max-attempts",
        "1",
        "--output",
        str(output),
    ]
    main(args)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    main(args)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads((tmp_path / "predictions.jsonl.manifest.json").read_text())
    assert manifest["upstream"]["commit"] == "f72f75d8c3eb842f2bbbab096a12206ff66e270f"

    judged = tmp_path / "judged.jsonl"
    main(
        [
            "judge",
            "--config",
            str(config),
            "--dataset",
            str(dataset),
            "--predictions",
            str(output),
            "--workers",
            "1",
            "--output",
            str(judged),
        ]
    )
    assert json.loads(judged.read_text(encoding="utf-8"))["correct"] is True


def test_network_backend_requires_explicit_paid_authorization(tmp_path):
    dataset = tmp_path / "questions.jsonl"
    write_jsonl(dataset, [{"question_id": "q1", "question": "Q"}])
    config = Path(__file__).parents[1] / "configs" / "example.json"
    with pytest.raises(RuntimeError, match="--allow-paid-apis"):
        main(
            [
                "run",
                "--config",
                str(config),
                "--dataset",
                str(dataset),
                "--method",
                "zero",
                "--threshold",
                "95",
                "--output",
                str(tmp_path / "never-created.jsonl"),
            ]
        )


def test_doctor_is_offline_and_supports_one_shared_model_key(monkeypatch, capsys):
    config = Path(__file__).parents[1] / "configs" / "paper-browseconf.json"
    monkeypatch.setenv("MODEL_API_KEY", "secret")
    monkeypatch.setenv("SERPER_KEY_ID", "secret")
    main(["doctor", "--config", str(config)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["network_calls_made"] is False
    assert payload["missing_required"] == []
    assert "secret" not in json.dumps(payload)
    assert payload["ready"] is False  # template URLs must still be replaced
