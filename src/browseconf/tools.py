"""Web tool adapters.

DeepResearch* adapters are modified from the observable behavior of
Alibaba-NLP/DeepResearch@f72f75d8c3eb842f2bbbab096a12206ff66e270f (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import ChatModel
from .parsing import parse_json_object
from .prompts import EXTRACTOR_PROMPT
from .schemas import ToolOutput, Usage


class Tool(Protocol):
    name: str

    def call(self, arguments: dict[str, Any]) -> str | ToolOutput: ...


@dataclass(slots=True)
class FileCache:
    root: Path
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with self._lock:
            temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            temp.replace(path)


def _json_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
    retries: int = 3,
) -> tuple[int, dict[str, Any] | str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                content_type = response.headers.get("Content-Type", "")
                return response.status, json.loads(body) if "json" in content_type else body
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 401, 403, 404}:
                break
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 4))
    raise RuntimeError(f"HTTP request failed: {url}: {last_error}")


@dataclass(slots=True)
class SerperSearchTool:
    """Google results through Serper, matching the released WebSailor scaffold."""

    api_key_env: str = "GOOGLE_SEARCH_KEY"
    country: str = "en"
    language: str | None = None
    top_k: int = 10
    workers: int = 3
    timeout_seconds: float = 30
    cache: FileCache | None = None
    name: str = "search"

    def _search_one(self, query: str) -> str:
        cache_key = json.dumps(
            {
                "query": query,
                "country": self.country,
                "language": self.language,
                "top_k": self.top_k,
            },
            sort_keys=True,
        )
        if self.cache and (cached := self.cache.get("search", cache_key)):
            return str(cached["rendered"])
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing search key in environment variable {self.api_key_env}")
        payload: dict[str, Any] = {"q": query, "num": self.top_k, "gl": self.country}
        if self.language:
            payload["hl"] = self.language
        _, response = _json_request(
            "https://google.serper.dev/search",
            method="POST",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            payload=payload,
            timeout=self.timeout_seconds,
        )
        if not isinstance(response, dict):
            raise TypeError("Serper returned a non-JSON response")
        results = response.get("organic") or []
        rendered_items: list[str] = []
        for index, item in enumerate(results[: self.top_k], start=1):
            title = item.get("title", "Untitled")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            date = f"\nDate published: {item['date']}" if item.get("date") else ""
            source = f"\nSource: {item['source']}" if item.get("source") else ""
            rendered_items.append(f"{index}. [{title}]({link}){date}{source}\n{snippet}")
        rendered = (
            f"A Google search for '{query}' found {len(rendered_items)} results:\n\n"
            "## Web Results\n" + "\n\n".join(rendered_items)
        )
        if self.cache:
            self.cache.put("search", cache_key, {"raw": response, "rendered": rendered})
        return rendered

    def call(self, arguments: dict[str, Any]) -> str:
        queries = arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(q, str) for q in queries)
        ):
            raise ValueError("search requires a non-empty string array 'query'")
        with ThreadPoolExecutor(max_workers=min(self.workers, len(queries))) as executor:
            return "\n=======\n".join(executor.map(self._search_one, queries))


@dataclass(slots=True)
class JinaReader:
    api_key_env: str = "JINA_API_KEY"
    max_chars: int = 150_000
    timeout_seconds: float = 30
    cache: FileCache | None = None

    def read(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("visit URL must start with http:// or https://")
        if self.cache and (cached := self.cache.get("jina", url)):
            return str(cached["content"])
        headers = {"Accept": "text/plain"}
        if key := os.getenv(self.api_key_env):
            headers["Authorization"] = f"Bearer {key}"
        _, response = _json_request(
            f"https://r.jina.ai/{url}", headers=headers, timeout=self.timeout_seconds
        )
        content = (
            response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        )
        content = content[: self.max_chars]
        if self.cache:
            self.cache.put("jina", url, {"content": content})
        return content


@dataclass(slots=True)
class VisitTool:
    reader: JinaReader
    summary_model: ChatModel
    workers: int = 3
    summary_temperature: float = 0.0
    summary_max_tokens: int = 4096
    name: str = "visit"

    def _visit_one(self, url: str, goal: str) -> ToolOutput:
        content = self.reader.read(url)
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        response = self.summary_model.generate(
            [{"role": "user", "content": prompt}],
            temperature=self.summary_temperature,
            max_tokens=self.summary_max_tokens,
        )
        try:
            parsed = parse_json_object(response.content)
            evidence = str(parsed.get("evidence", ""))
            summary = str(parsed.get("summary", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            evidence = response.content
            summary = "Summary model did not return valid JSON; raw output preserved."
        rendered = (
            f"The useful information in {url} for user goal {goal} is as follows:\n\n"
            f"Evidence in page:\n{evidence}\n\nSummary:\n{summary}\n"
        )
        return ToolOutput(content=rendered, usage=response.usage, metadata={"url": url})

    def call(self, arguments: dict[str, Any]) -> str:
        urls = arguments.get("url")
        goal = arguments.get("goal")
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) for url in urls):
            raise ValueError("visit requires a URL string or non-empty URL array")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("visit requires a non-empty string 'goal'")
        with ThreadPoolExecutor(max_workers=min(self.workers, len(urls))) as executor:
            outputs = list(executor.map(lambda url: self._visit_one(url, goal), urls))
        usage = Usage()
        for output in outputs:
            usage.add(output.usage)
        return ToolOutput(
            content="\n=======\n".join(output.content for output in outputs),
            usage=usage,
            metadata={"urls": urls, "summary_calls": len(outputs)},
        )


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


@dataclass(slots=True)
class DeepResearchSearchAdapter:
    """Faithful, dependency-free boundary adapter for DeepResearch ``Search``.

    The request locale and rendering follow inference/tool_search.py at the pinned
    upstream commit.  Batch queries are deliberately sequential, as in that file.
    """

    api_key_env: str = "SERPER_KEY_ID"
    endpoint: str = "https://google.serper.dev/search"
    top_k: int = 10
    timeout_seconds: float = 30
    retries: int = 5
    cache: FileCache | None = None
    name: str = "search"

    def _search_one(self, query: str) -> str:
        chinese = _contains_cjk(query)
        payload = {
            "q": query,
            "location": "China" if chinese else "United States",
            "gl": "cn" if chinese else "us",
            "hl": "zh-cn" if chinese else "en",
        }
        cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if self.cache and (cached := self.cache.get("deepresearch-search", cache_key)):
            return str(cached["rendered"])
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing search key in environment variable {self.api_key_env}")
        try:
            _, response = _json_request(
                self.endpoint,
                method="POST",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                payload=payload,
                timeout=self.timeout_seconds,
                retries=self.retries,
            )
        except RuntimeError:
            return "Google search Timeout, return None, Please try again later."
        if not isinstance(response, dict) or "organic" not in response:
            return f"No results found for '{query}'. Try with a more general query."
        items: list[str] = []
        for index, page in enumerate(response["organic"][: self.top_k], start=1):
            date = f"\nDate published: {page['date']}" if page.get("date") else ""
            source = f"\nSource: {page['source']}" if page.get("source") else ""
            snippet = f"\n{page['snippet']}" if page.get("snippet") else ""
            rendered = (
                f"{index}. [{page['title']}]({page['link']}){date}{source}\n{snippet}"
            ).replace("Your browser can't play this video.", "")
            items.append(rendered)
        result = (
            f"A Google search for '{query}' found {len(items)} results:\n\n"
            "## Web Results\n" + "\n\n".join(items)
        )
        if self.cache:
            self.cache.put("deepresearch-search", cache_key, {"raw": response, "rendered": result})
        return result

    def call(self, arguments: dict[str, Any]) -> str:
        try:
            queries = arguments["query"]
        except (KeyError, TypeError):
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        if isinstance(queries, str):
            return self._search_one(queries)
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(query, str) for query in queries)
        ):
            raise ValueError("search 'query' must be a string or a non-empty string array")
        return "\n=======\n".join(self._search_one(query) for query in queries)


@dataclass(slots=True)
class DeepResearchJinaReader:
    """Jina reader behavior adapted from DeepResearch inference/tool_visit.py."""

    api_key_env: str = "JINA_API_KEYS"
    endpoint: str = "https://r.jina.ai/"
    timeout_seconds: float = 50
    request_retries: int = 3
    outer_retries: int = 8
    retry_sleep_seconds: float = 0.5
    cache: FileCache | None = None

    def read(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("visit URL must start with http:// or https://")
        if self.cache and (cached := self.cache.get("deepresearch-jina", url)):
            return str(cached["content"])
        key = os.getenv(self.api_key_env, "")
        for _ in range(self.outer_retries):
            for attempt in range(self.request_retries):
                try:
                    _, response = _json_request(
                        f"{self.endpoint}{url}",
                        headers={"Authorization": f"Bearer {key}"},
                        timeout=self.timeout_seconds,
                        retries=1,
                    )
                    content = (
                        response
                        if isinstance(response, str)
                        else json.dumps(response, ensure_ascii=False)
                    )
                    if content and content != "[visit] Empty content.":
                        if self.cache:
                            self.cache.put("deepresearch-jina", url, {"content": content})
                        return content
                except RuntimeError:
                    pass
                if attempt + 1 < self.request_retries and self.retry_sleep_seconds:
                    time.sleep(self.retry_sleep_seconds)
        return "[visit] Failed to read page."


def _truncate_to_token_budget(text: str, max_tokens: int) -> tuple[str, str]:
    """Use upstream cl100k_base truncation when tiktoken is installed."""

    try:
        import tiktoken  # type: ignore[import-not-found]

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:max_tokens]), "cl100k_base"
    except ImportError:
        # Dependency-free fallback. CJK is approximately one token/character;
        # predominantly ASCII pages are conservatively capped at 4 chars/token.
        cap = max_tokens if sum(ord(c) > 127 for c in text) > len(text) // 4 else max_tokens * 4
        return text[:cap], "heuristic"


@dataclass(slots=True)
class DeepResearchVisitAdapter:
    """DeepResearch visit/extraction contract with injectable model and reader."""

    reader: DeepResearchJinaReader
    summary_model: ChatModel
    summary_temperature: float = 0.7
    summary_max_tokens: int | None = None
    max_page_tokens: int = 95_000
    summary_retries: int = 3
    parse_retries: int = 3
    batch_timeout_seconds: float = 900
    name: str = "visit"

    @staticmethod
    def _failure(url: str, goal: str) -> str:
        return (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            "Evidence in page: \nThe provided webpage content could not be accessed. "
            "Please check the URL or file format.\n\n"
            "Summary: \nThe webpage content could not be processed, and therefore, "
            "no information is available.\n\n"
        )

    def _model_call(self, messages: list[dict[str, str]]) -> tuple[str, Usage]:
        try:
            response = self.summary_model.generate(
                messages,
                temperature=self.summary_temperature,
                max_tokens=self.summary_max_tokens,
            )
            return response.content, response.usage
        except Exception:  # noqa: BLE001 - official adapter converts failures to empty output
            return "", Usage()

    def _visit_one(self, url: str, goal: str) -> ToolOutput:
        content = self.reader.read(url)
        if not content or content.startswith("[visit] Failed to read page"):
            return ToolOutput(
                content=self._failure(url, goal), metadata={"url": url, "read_failed": True}
            )
        content, truncation = _truncate_to_token_budget(content, self.max_page_tokens)
        usage = Usage()

        def messages_for(page: str) -> list[dict[str, str]]:
            return [
                {
                    "role": "user",
                    "content": EXTRACTOR_PROMPT.format(webpage_content=page, goal=goal),
                }
            ]

        messages = messages_for(content)
        raw, call_usage = self._model_call(messages)
        usage.add(call_usage)
        for retry in range(self.summary_retries + 1):
            if len(raw) >= 10:
                break
            content = (
                content[: int(0.7 * len(content))]
                if retry < self.summary_retries
                else content[:25_000]
            )
            messages = messages_for(content)
            raw, call_usage = self._model_call(messages)
            usage.add(call_usage)

        parsed: dict[str, Any] | None = None
        for attempt in range(self.parse_retries + 1):
            try:
                parsed = parse_json_object(raw.replace("```json", "").replace("```", "").strip())
                if "evidence" not in parsed or "summary" not in parsed:
                    raise ValueError("missing evidence or summary")
                break
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
                if attempt < self.parse_retries:
                    raw, call_usage = self._model_call(messages)
                    usage.add(call_usage)
        if parsed is None:
            rendered = self._failure(url, goal)
        else:
            rendered = (
                f"The useful information in {url} for user goal {goal} as follows: \n\n"
                f"Evidence in page: \n{parsed['evidence']}\n\n"
                f"Summary: \n{parsed['summary']}\n\n"
            )
        return ToolOutput(
            content=rendered,
            usage=usage,
            metadata={"url": url, "truncation": truncation, "parse_failed": parsed is None},
        )

    def call(self, arguments: dict[str, Any]) -> ToolOutput | str:
        try:
            urls = arguments["url"]
            goal = arguments["goal"]
        except (KeyError, TypeError):
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls or not all(isinstance(url, str) for url in urls):
            raise ValueError("visit 'url' must be a string or a non-empty string array")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("visit requires a non-empty string 'goal'")
        started = time.monotonic()
        outputs: list[ToolOutput] = []
        for url in urls:
            if time.monotonic() - started > self.batch_timeout_seconds:
                outputs.append(
                    ToolOutput(
                        content=self._failure(url, goal),
                        metadata={"url": url, "batch_timeout": True},
                    )
                )
                continue
            try:
                outputs.append(self._visit_one(url, goal))
            except Exception as exc:  # noqa: BLE001 - preserve per-URL batch progress
                outputs.append(
                    ToolOutput(
                        content=f"Error fetching {url}: {exc}",
                        metadata={"url": url, "error": str(exc)},
                    )
                )
        usage = Usage()
        for output in outputs:
            usage.add(output.usage)
        return ToolOutput(
            content="\n=======\n".join(output.content for output in outputs).strip(),
            usage=usage,
            metadata={
                "urls": urls,
                "summary_calls": len(outputs),
                "sequential": True,
                "results": [output.metadata for output in outputs],
            },
        )


@dataclass(slots=True)
class ToolRouter:
    tools: dict[str, Tool]

    @classmethod
    def from_tools(cls, *tools: Tool) -> ToolRouter:
        return cls({tool.name: tool for tool in tools})

    def call(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        output = self.tools[name].call(arguments)
        return output if isinstance(output, ToolOutput) else ToolOutput(content=output)


@dataclass(slots=True)
class StaticTool:
    name: str
    result: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def call(self, arguments: dict[str, Any]) -> str:
        self.calls.append(arguments)
        return self.result
