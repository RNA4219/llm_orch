"""READMEのJSONスキーマ検証テスト。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.orch.types import ChatRequest

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
SCHEMAS = {"ChatRequest": ChatRequest}


def iter_schema_blocks(readme_text: str) -> Iterator[Tuple[str, str]]:
    pattern = re.compile(
        r"<!--\s*schema:\s*(?P<name>[A-Za-z0-9_]+)\s*-->\s*```json\n(?P<body>.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(readme_text):
        yield match.group("name"), match.group("body")


@pytest.mark.parametrize("schema_name, raw_json", tuple(iter_schema_blocks(README_PATH.read_text("utf-8"))))
def test_readme_json_blocks(schema_name: str, raw_json: str) -> None:
    assert schema_name in SCHEMAS, f"未知のスキーマ: {schema_name}"
    payload = json.loads(raw_json)
    SCHEMAS[schema_name].model_validate(payload)


def test_readme_has_schema_blocks() -> None:
    blocks = tuple(iter_schema_blocks(README_PATH.read_text("utf-8")))
    assert blocks, "READMEにschemaタグ付きJSONコードブロックがありません"
