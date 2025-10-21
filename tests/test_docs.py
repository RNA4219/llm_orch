"""READMEのJSONスキーマ検証テスト。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
                {"role": "system", "content": "You are a routing assistant."},
                {
                    "role": "user",
                    "content": "最新レコメンドの候補を3つ提案して。",
                },
            ],
            "temperature": 0.3,
            "stream": False,
        },
        "ChatRequestStreamPython": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "SSEで自己紹介を短く返して。",
                },
            ],
            "temperature": 0.7,
            "max_tokens": 256,
            "stream": True,
        },
        "ChatRequestStreamJavaScript": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You respond with JSON fragments."},
                {
                    "role": "user",
                    "content": "進捗報告テンプレートをストリームで送って。",
                },
            ],
            "temperature": 0.5,
            "max_tokens": 200,
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
