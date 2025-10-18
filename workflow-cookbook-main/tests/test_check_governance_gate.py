import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.ci.check_governance_gate import (
    find_forbidden_matches,
    load_forbidden_patterns,
    main,
    validate_priority_score,
)


@pytest.mark.parametrize(
    "changed_paths, patterns, expected",
    [
        ("""core/schema/model.yaml\ndocs/guide.md""".splitlines(), ["/core/schema/**"], ["core/schema/model.yaml"]),
        (
            """core/schema/nested/file.yaml""".splitlines(),
            ["/core/schema/**"],
            ["core/schema/nested/file.yaml"],
        ),
        (
            """docs/sub/file.md""".splitlines(),
            ["./docs/**"],
            ["docs/sub/file.md"],
        ),
        ("""docs/readme.md\nops/runbook.md""".splitlines(), ["/core/schema/**"], []),
        (
            """auth/service.py\ncore/schema/definitions.yml""".splitlines(),
            ["/auth/**", "/core/schema/**"],
            ["auth/service.py", "core/schema/definitions.yml"],
        ),
    ],
)
def test_find_forbidden_matches(changed_paths, patterns, expected):
    normalized = [pattern.lstrip("/") for pattern in patterns]
    assert find_forbidden_matches(changed_paths, normalized) == expected


@pytest.mark.parametrize(
    "body",
    [
        "Priority Score: 5 / 安全性強化",
        "Priority Score: 1 / 即応性向上",
    ],
)
def test_validate_priority_score_accepts_valid_format(body):
    assert validate_priority_score(body) is True


@pytest.mark.parametrize(
    "body",
    [
        "Priority Score: 3",
        "Priority Score: / 理由",
        "Priority Score: abc / 理由",
        "Priority Score: <!-- 例: 5 / prioritization.yaml#phase1 -->",
        "priority score: 3",
        "",
        None,
    ],
)
def test_validate_priority_score_rejects_invalid_format(body):
    assert validate_priority_score(body) is False


def test_load_forbidden_patterns(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        """
self_modification:
  forbidden_paths:
    - "/core/schema/**"
    - '/auth/**'
  require_human_approval:
    - "/governance/**"
"""
    )

    assert load_forbidden_patterns(policy) == ["core/schema/**", "auth/**"]


@pytest.mark.parametrize(
    "item, expected",
    [
        ("- /infra/**  # コメント", ["infra/**"]),
        ("- '/secure/**'  # コメント", ["secure/**"]),
        ("- \"/quoted/**\"  # コメント", ["quoted/**"]),
        ("- \"#literal\"  # コメント", ["#literal"]),
        ("- # コメントのみ", []),
    ],
)
def test_load_forbidden_patterns_supports_inline_comments(tmp_path, item, expected):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            [
                "self_modification:",
                "  forbidden_paths:",
                f"    {item}",
            ]
        )
    )

    assert load_forbidden_patterns(policy) == expected


def test_load_forbidden_patterns_preserves_hash_within_quotes(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            [
                "self_modification:",
                "  forbidden_paths:",
                "    - '/path#literal'  # trailing comment",
            ]
        )
    )

    assert load_forbidden_patterns(policy) == ["path#literal"]


def test_main_returns_error_when_priority_invalid(monkeypatch, tmp_path, capsys):
    event_file = tmp_path / "event.json"
    event_file.write_text("""{"pull_request": {"body": "invalid"}}""")

    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.load_forbidden_patterns", lambda _: []
    )
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.get_changed_paths", lambda _: []
    )
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.find_forbidden_matches",
        lambda _paths, _patterns: [],
    )
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.validate_priority_score", lambda body: False
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "priority score validation failed" in captured.err.lower()


def test_main_succeeds_when_priority_valid(monkeypatch, tmp_path, capsys):
    event_file = tmp_path / "event.json"
    event_file.write_text("""{"pull_request": {"body": "valid"}}""")

    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_file))
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.load_forbidden_patterns", lambda _: []
    )
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.get_changed_paths", lambda _: []
    )
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.find_forbidden_matches",
        lambda _paths, _patterns: [],
    )
    monkeypatch.setattr(
        "tools.ci.check_governance_gate.validate_priority_score", lambda body: True
    )

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
