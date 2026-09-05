import json
from dataclasses import dataclass

from browseconf import tools
from browseconf.judge import DeepResearchJudgeAdapter
from browseconf.models import ScriptedModel
from browseconf.runner import build_agent, build_judge
from browseconf.tools import (
    DeepResearchJinaReader,
    DeepResearchSearchAdapter,
    DeepResearchVisitAdapter,
)


def test_official_search_request_and_rendering(monkeypatch):
    calls = []

    def fake_request(url, **kwargs):
        calls.append((url, kwargs))
        return 200, {
            "organic": [
                {
                    "title": "T",
                    "link": "https://example.test",
                    "snippet": "S",
                    "date": "2025",
                    "source": "Source",
                }
            ]
        }

    monkeypatch.setenv("SERPER_TEST", "secret")
    monkeypatch.setattr(tools, "_json_request", fake_request)
    adapter = DeepResearchSearchAdapter(api_key_env="SERPER_TEST")
    result = adapter.call({"query": "english query"})
    assert calls[0][0] == "https://google.serper.dev/search"
    assert calls[0][1]["payload"] == {
        "q": "english query",
        "location": "United States",
        "gl": "us",
        "hl": "en",
    }
    assert "num" not in calls[0][1]["payload"]
    assert "Date published: 2025" in result
    assert "Source: Source" in result


def test_official_search_batch_is_ordered_and_chinese_localized(monkeypatch):
    payloads = []

    def fake_request(_url, **kwargs):
        payloads.append(kwargs["payload"])
        query = kwargs["payload"]["q"]
        return 200, {"organic": [{"title": query, "link": "https://x", "snippet": "s"}]}

    monkeypatch.setenv("SERPER_TEST", "secret")
    monkeypatch.setattr(tools, "_json_request", fake_request)
    result = DeepResearchSearchAdapter(api_key_env="SERPER_TEST").call(
        {"query": ["中文查询", "second"]}
    )
    assert [payload["q"] for payload in payloads] == ["中文查询", "second"]
    assert payloads[0]["location"] == "China"
    assert payloads[0]["hl"] == "zh-cn"
    assert result.index("中文查询") < result.index("second")


def test_jina_failure_timeout_empty_and_retry(monkeypatch):
    responses = iter([RuntimeError("timeout"), (200, ""), (200, "page")])

    def fake_request(*_args, **_kwargs):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(tools, "_json_request", fake_request)
    reader = DeepResearchJinaReader(request_retries=3, outer_retries=1, retry_sleep_seconds=0)
    assert reader.read("https://example.test") == "page"

    monkeypatch.setattr(
        tools, "_json_request", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError())
    )
    failed = DeepResearchJinaReader(request_retries=2, outer_retries=2, retry_sleep_seconds=0)
    assert failed.read("https://example.test") == "[visit] Failed to read page."


@dataclass
class FakeReader:
    pages: dict[str, str]

    def read(self, url):
        return self.pages[url]


def test_visit_single_multi_and_summary_json():
    model = ScriptedModel(
        [
            '{"rational":"r","evidence":"e1","summary":"s1"}',
            '{"rational":"r","evidence":"e2","summary":"s2"}',
        ]
    )
    adapter = DeepResearchVisitAdapter(
        reader=FakeReader({"https://a": "page a", "https://b": "page b"}),  # type: ignore[arg-type]
        summary_model=model,
    )
    output = adapter.call({"url": ["https://a", "https://b"], "goal": "goal"})
    assert "e1" in output.content and "e2" in output.content
    assert output.metadata["sequential"] is True
    assert len(model.calls) == 2
    assert model.call_options[0]["temperature"] == 0.7


def test_visit_failed_page_and_parse_failure_are_stable():
    failed_reader = FakeReader({"https://a": "[visit] Failed to read page."})
    failed = DeepResearchVisitAdapter(
        reader=failed_reader,
        summary_model=ScriptedModel([]),  # type: ignore[arg-type]
    ).call({"url": "https://a", "goal": "g"})
    assert "could not be accessed" in failed.content

    model = ScriptedModel(["not json"] * 4)
    parsed = DeepResearchVisitAdapter(
        reader=FakeReader({"https://a": "page"}),  # type: ignore[arg-type]
        summary_model=model,
        summary_retries=0,
        parse_retries=3,
    ).call({"url": "https://a", "goal": "g"})
    assert "could not be processed" in parsed.content
    assert parsed.metadata["urls"] == ["https://a"]


def test_visit_long_page_is_truncated_before_summary():
    model = ScriptedModel(['{"evidence":"e","summary":"s"}'])
    adapter = DeepResearchVisitAdapter(
        reader=FakeReader({"https://a": "x" * 1000}),  # type: ignore[arg-type]
        summary_model=model,
        max_page_tokens=10,
    )
    adapter.call({"url": "https://a", "goal": "g"})
    assert "x" * 1000 not in model.calls[0][0]["content"]


def test_judge_correct_incorrect_parse_failure_and_api_failure():
    model = ScriptedModel(
        [
            json.dumps({"correct": "yes"}),
            json.dumps({"correct": "no"}),
            "unparseable",
        ]
    )
    judge = DeepResearchJudgeAdapter(model, attempts=1, retry_delay_seconds=0)
    common = {"question_id": "q", "question": "Q", "correct_answer": "A", "response": "A"}
    assert judge.grade(**common).correct is True
    assert judge.grade(**common).correct is False
    assert judge.grade(**common).error == "judge_parse_failure"
    assert model.call_options[0]["response_format"]["type"] == "json_schema"

    class BrokenModel:
        model_id = "broken"

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("api down")

    failure = DeepResearchJudgeAdapter(
        BrokenModel(),
        attempts=2,
        retry_delay_seconds=0,  # type: ignore[arg-type]
    ).grade(**common)
    assert failure.correct is None
    assert failure.error == "api down"


def test_model_credentials_and_endpoints_are_independent():
    config = {
        "backend": "official-deepresearch",
        "agent_model": {"model_id": "agent", "base_url": "https://agent", "api_key_env": "A"},
        "summary_model": {
            "model_id": "summary",
            "base_url": "https://summary",
            "api_key_env": "S",
        },
        "judge_model": {"model_id": "judge", "base_url": "https://judge", "api_key_env": "J"},
    }
    agent, summary = build_agent(config)
    judge = build_judge(config)
    assert agent.model.api_key_env == "A" and agent.model.base_url == "https://agent"
    assert summary.api_key_env == "S" and summary.base_url == "https://summary"
    assert judge.model.api_key_env == "J" and judge.model.base_url == "https://judge"
    assert len({agent.model.api_key_env, summary.api_key_env, judge.model.api_key_env}) == 3


def test_offline_backend_needs_no_credentials():
    config = {
        "backend": "offline-mock",
        "offline": {
            "agent_responses": ["<answer>Answer: a\nConfidence: 99</answer>"],
            "summary_responses": [],
        },
    }
    agent, _ = build_agent(config)
    attempt = agent.run(question_id="q", question="Q", attempt_index=1)
    assert attempt.answer == "a"


def test_paper_profile_locks_published_agent_settings():
    config = {
        "backend": "paper-browseconf",
        "agent_model": {"model_id": "gpt-oss-120b", "api_key_env": "ONE"},
        "summary_model": {"model_id": "gpt-oss-120b", "api_key_env": "ONE"},
        "judge_model": {"model_id": "gpt-4o-2024-08-06", "api_key_env": "ONE"},
        "agent": {"temperature": 0.6, "top_p": 0.95, "max_context_tokens": 131072},
    }
    agent, summary = build_agent(config)
    assert agent.protocol == "paper-browseconf"
    assert agent.max_context_tokens == 131_072
    assert agent.presence_penalty is None
    assert agent.logprobs is None
    assert agent.temperature == 0.6
    assert agent.top_p == 0.95
    assert summary.model_id == "gpt-oss-120b"
