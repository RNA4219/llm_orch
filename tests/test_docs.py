"""READMEのJSONスキーマ検証テスト。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest

from src.orch.types import ChatRequest

def iter_schema_blocks(readme_text: str) -> Iterator[Tuple[str, str]]:
    pattern = re.compile(
        r"<!--\s*schema:\s*(?P<name>[A-Za-z0-9_]+)\s*-->\s*```json\n(?P<body>.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(readme_text):
        yield match.group("name"), match.group("body")


README_PATH = Path(__file__).resolve().parent.parent / "README.md"

README_TEXT = README_PATH.read_text("utf-8")
README_SCHEMA_BLOCKS: Tuple[Tuple[str, str], ...] = tuple(
    iter_schema_blocks(README_TEXT)
)

SCHEMAS = {
    "ChatRequest": ChatRequest,
    "ChatRequestStickyCurl": ChatRequest,
    "ChatRequestStreamPython": ChatRequest,
    "ChatRequestStreamJavaScript": ChatRequest,
}


@pytest.fixture(scope="module")
def schema_payloads() -> Dict[str, Dict[str, Any]]:
    return {name: json.loads(body) for name, body in README_SCHEMA_BLOCKS}


@pytest.fixture(scope="module")
def expected_readme_examples() -> Dict[str, Dict[str, Any]]:
    return {
        "ChatRequestStickyCurl": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You triage loyalty checkout recommendations.",
                },
                {
                    "role": "user",
                    "content": "カート checkout-42 に向けたレコメンドを3件まとめて。",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": False,
        },
        "ChatRequestStreamPython": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You summarize patch notes in friendly Japanese.",
                },
                {
                    "role": "user",
                    "content": "以下のdiffを要約して: add webhook retry metric",
                },
            ],
            "temperature": 0.6,
            "max_tokens": 192,
            "stream": True,
        },
        "ChatRequestStreamJavaScript": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You emit status updates as JSON fragments.",
                },
                {
                    "role": "user",
                    "content": "スプリント3の進捗レポート雛形をストリームで。",
                },
            ],
            "temperature": 0.4,
            "max_tokens": 220,
            "stream": True,
        },
    }


@pytest.mark.parametrize("schema_name, raw_json", README_SCHEMA_BLOCKS)
def test_readme_json_blocks(schema_name: str, raw_json: str) -> None:
    assert schema_name in SCHEMAS, f"未知のスキーマ: {schema_name}"
    payload = json.loads(raw_json)
    SCHEMAS[schema_name].model_validate(payload)


def test_readme_has_schema_blocks() -> None:
    assert README_SCHEMA_BLOCKS, "READMEにschemaタグ付きJSONコードブロックがありません"


def test_readme_examples_cover_expected_cases(
    schema_payloads: Dict[str, Dict[str, Any]],
    expected_readme_examples: Dict[str, Dict[str, Any]],
) -> None:
    for name, expected in expected_readme_examples.items():
        assert name in schema_payloads, f"READMEに{name}の例がありません"
        assert (
            schema_payloads[name] == expected
        ), f"READMEの{name}例が期待するJSONと一致しません"
