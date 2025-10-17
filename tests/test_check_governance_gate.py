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
        "This PR fixes something important.",
        "Priority score: 5 / 安全性強化",  # casing mismatch
        "Priority Score:",
        "Priority Score:  ",
        "Priority Score: / 理由",
        "Priority Score:  /  理由",
        "Priority Score: five / 理由",
        "Priority Score: 5 /   ",
        "Priority Score: <!-- 例: 5 / prioritization.yaml#phase1 -->",
    ],
)
def test_validate_priority_score_returns_false_when_missing_or_invalid(body: str | None) -> None:
    assert validate_priority_score(body) is False


@pytest.mark.parametrize(
    "body",
    [
        "Priority Score: 5 / 安全性強化",
        "Some context before\nPriority Score: 3 / prioritization.yaml#phase1",
        "* Priority Score: 1 / 応急対応",
    ],
)
def test_validate_priority_score_returns_true_for_valid_entries(body: str) -> None:
    assert validate_priority_score(body) is True
