from __future__ import annotations

import datetime as dt
import hashlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from . import prompts
from .schemas import to_dict
from .storage import atomic_write_json

_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "password",
    "secret",
    "access_token",
    "auth_token",
    "authorization",
    "bearer",
)
DEEPRESEARCH_COMMIT = "f72f75d8c3eb842f2bbbab096a12206ff66e270f"
DEEPRESEARCH_REPOSITORY = "https://github.com/Alibaba-NLP/DeepResearch"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(value: Any, key: str = "") -> Any:
    """Remove literal credentials while retaining environment-variable names."""
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS) and not lowered.endswith("_env"):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(item_key): sanitize(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item, key) for item in value]
    return to_dict(value)


def prompt_hashes() -> dict[str, str]:
    names = (
        "BASE_SYSTEM_PROMPT",
        "CONFIDENCE_SYSTEM_PROMPT",
        "PAPER_TOOL_PROTOCOL",
        "DEEPRESEARCH_SYSTEM_PROMPT",
        "TOOLS_AND_PROTOCOL",
        "SUMMARY_QUESTION",
        "NEG_QUESTION",
        "EXTRACTOR_PROMPT",
        "INITIAL_SUMMARY_PROMPT",
        "SUBSEQUENT_SUMMARY_PROMPT",
        "BROWSECOMP_JUDGE_PROMPT",
        "ANSWER_EQUIVALENCE_PROMPT",
    )
    return {
        name: hashlib.sha256(getattr(prompts, name).encode("utf-8")).hexdigest() for name in names
    }


def write_manifest(
    output_path: str | Path,
    *,
    command: str,
    arguments: dict[str, Any],
    config: dict[str, Any] | None = None,
    inputs: list[str | Path] | None = None,
) -> Path:
    output = Path(output_path)
    hashed_inputs: dict[str, dict[str, Any]] = {}
    for item in inputs or []:
        path = Path(item)
        if path.exists() and path.is_file():
            hashed_inputs[str(path.resolve())] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    try:
        package_version = version("browseconf-repro")
    except PackageNotFoundError:
        package_version = "uninstalled"
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "command": command,
        "arguments": sanitize(arguments),
        "config": sanitize(config) if config is not None else None,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output) if output.exists() else None,
        "inputs": hashed_inputs,
        "prompt_sha256": prompt_hashes(),
        "runtime": {
            "package_version": package_version,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "upstream": {
            "repository": DEEPRESEARCH_REPOSITORY,
            "commit": DEEPRESEARCH_COMMIT,
            "license": "Apache-2.0",
            "integration": "compatibility-adapter",
        },
    }
    destination = Path(f"{output}.manifest.json")
    atomic_write_json(destination, payload)
    return destination
