from dataclasses import dataclass

from browseconf.models import ScriptedModel
from browseconf.policies import BrowseConfRunner, TrajectorySummarizer
from browseconf.schemas import Attempt


@dataclass
class FakeAttemptRunner:
    scripted: list[tuple[str | None, int, str]]
    model: ScriptedModel

    def run(self, *, question_id: str, question: str, attempt_index: int) -> Attempt:
        answer, confidence, stop_reason = self.scripted.pop(0)
        return Attempt(
            question_id=question_id,
            attempt_index=attempt_index,
            answer=answer,
            confidence=confidence,
            raw_output="",
            messages=[{"role": "user", "content": question}],
            stop_reason=stop_reason,
        )


def test_zero_stops_at_threshold():
    fake = FakeAttemptRunner(
        [("a", 40, "answer"), ("b", 95, "answer"), ("c", 99, "answer")],
        ScriptedModel([]),
    )
    result = BrowseConfRunner(fake, method="zero", threshold=90, max_attempts=3).run("q", "Q")
    assert result.final_answer == "b"
    assert result.attempt_count == 2
    assert result.stop_reason == "threshold"


def test_budget_exhaustion_selects_earliest_max_confidence():
    fake = FakeAttemptRunner(
        [("a", 80, "answer"), ("b", 80, "answer"), ("c", 70, "answer")],
        ScriptedModel([]),
    )
    result = BrowseConfRunner(fake, method="zero", threshold=90, max_attempts=3).run("q", "Q")
    assert result.final_answer == "a"
    assert result.selected_attempt_index == 1


def test_neg_ignores_overflow_without_answer_and_passes_previous_answer():
    fake = FakeAttemptRunner(
        [(None, -1, "context_overflow"), ("wrong", 20, "answer"), ("right", 99, "answer")],
        ScriptedModel([]),
    )
    result = BrowseConfRunner(fake, method="neg", threshold=90, max_attempts=3).run("q", "Q")
    assert "<incorrect_answers>" not in result.attempts[1].messages[0]["content"]
    assert "- wrong" in result.attempts[2].messages[0]["content"]


def test_summary_carries_summary_forward():
    fake = FakeAttemptRunner(
        [("wrong", 20, "answer"), ("right", 99, "answer")],
        ScriptedModel([]),
    )
    summary_model = ScriptedModel(["summary-v1"])
    result = BrowseConfRunner(
        fake,
        method="summary",
        threshold=90,
        max_attempts=2,
        summarizer=TrajectorySummarizer(summary_model),
    ).run("q", "Q")
    assert "summary-v1" in result.attempts[1].messages[0]["content"]
    assert len(result.summaries) == 1


def test_summary_does_not_pay_for_unused_final_summary():
    fake = FakeAttemptRunner(
        [("wrong-a", 20, "answer"), ("wrong-b", 30, "answer")],
        ScriptedModel([]),
    )
    summary_model = ScriptedModel(["summary-v1"])
    result = BrowseConfRunner(
        fake,
        method="summary",
        threshold=90,
        max_attempts=2,
        summarizer=TrajectorySummarizer(summary_model),
    ).run("q", "Q")
    assert len(result.summaries) == 1
    assert len(summary_model.calls) == 1
