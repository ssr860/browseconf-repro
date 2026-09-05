import json

from browseconf.manifest import prompt_hashes, sanitize, sha256_file, write_manifest


def test_manifest_hashes_and_redacts_secrets(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"ok": true}', encoding="utf-8")
    source = tmp_path / "input.jsonl"
    source.write_text('{"id": 1}\n', encoding="utf-8")
    path = write_manifest(
        output,
        command="test",
        arguments={"threshold": 90},
        config={
            "api_key": "literal-secret",
            "api_key_env": "MODEL_API_KEY",
            "max_tokens": 8192,
        },
        inputs=[source],
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["config"]["api_key"] == "<redacted>"
    assert manifest["config"]["api_key_env"] == "MODEL_API_KEY"
    assert manifest["config"]["max_tokens"] == 8192
    assert manifest["output_sha256"] == sha256_file(output)
    assert manifest["inputs"][str(source.resolve())]["sha256"] == sha256_file(source)
    assert len(prompt_hashes()["CONFIDENCE_SYSTEM_PROMPT"]) == 64


def test_sanitize_nested_secret():
    assert sanitize({"nested": {"password": "x"}})["nested"]["password"] == "<redacted>"
