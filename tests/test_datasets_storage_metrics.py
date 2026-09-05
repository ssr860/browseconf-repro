import base64
import json

from browseconf.datasets import decrypt, derive_key, load_browsecomp_csv, load_jsonl
from browseconf.metrics import result_report
from browseconf.storage import JsonlStore, read_jsonl


def encrypt(text: str, password: str) -> str:
    raw = text.encode()
    key = derive_key(password, len(raw))
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, key))).decode()


def test_browsecomp_decryption_and_loader(tmp_path):
    password = "canary"
    problem = encrypt("question", password)
    answer = encrypt("answer", password)
    path = tmp_path / "data.csv"
    path.write_text(f"problem,answer,canary\n{problem},{answer},{password}\n", encoding="utf-8")
    rows = load_browsecomp_csv(path)
    assert rows[0].question == "question"
    assert rows[0].answer == "answer"
    assert decrypt(answer, password) == "answer"


def test_jsonl_and_idempotency_keys(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"id": "1", "question": "q", "answer": "a"}) + "\n")
    assert load_jsonl(path)[0].question_id == "1"
    output = tmp_path / "out.jsonl"
    store = JsonlStore(output)
    store.append({"question_id": "1", "x": 2})
    assert store.keys() == {"1"}
    assert read_jsonl(output)[0]["x"] == 2


def test_result_report():
    predictions = [
        {
            "question_id": "1",
            "final_confidence": 90,
            "attempt_count": 1,
            "stop_reason": "threshold",
        },
        {
            "question_id": "2",
            "final_confidence": 10,
            "attempt_count": 2,
            "stop_reason": "budget_exhausted",
        },
    ]
    judgements = [
        {"question_id": "1", "correct": True},
        {"question_id": "2", "correct": False},
    ]
    report = result_report(predictions, judgements)
    assert report["accuracy"] == 0.5
    assert report["avg_attempts"] == 1.5
    assert report["confidence"]["auroc"] == 1.0
