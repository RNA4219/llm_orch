from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1] / "workflow-cookbook-main"))

from tools.ci.check_governance_gate import validate_priority_score


@pytest.mark.parametrize(
    "body",
    [
        None,
        "",
        "本文には Priority Score がありません",
    ],
)
def test_validate_priority_score_returns_false_when_section_missing(body: str | None) -> None:
    assert validate_priority_score(body) is False


@pytest.mark.parametrize(
    "body",
    [
        "Priority Score: 3",
        "Priority Score: / 根拠なし",
        "Priority Score: abc / 理由",
        "Priority Score: 0 / 理由",
        "Priority Score: 6 / 理由",
        "Priority Score: 2 / <!-- placeholder -->",
    ],
)
def test_validate_priority_score_returns_false_when_format_invalid(body: str) -> None:
    assert validate_priority_score(body) is False


def test_validate_priority_score_returns_true_when_format_valid() -> None:
    body = """
    ## Summary
    - 変更内容

    Priority Score: 4 / prioritization.yaml#phase2 に基づき対応
    """.strip()

    assert validate_priority_score(body) is True
