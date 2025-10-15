import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze


@pytest.fixture
def tmp_paths(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    reports_dir = tmp_path / "reports"
    logs_dir.mkdir()
    reports_dir.mkdir()
    return logs_dir, reports_dir


def test_analyze_main_generates_report(tmp_paths) -> None:
    logs_dir, reports_dir = tmp_paths
    log_path = logs_dir / "test.jsonl"
    log_path.write_text(json.dumps({
        "name": "sample::test_case",
        "duration_ms": 123,
        "status": "pass",
    }) + "\n", encoding="utf-8")

    report_path = reports_dir / "today.md"
    issue_path = reports_dir / "issue_suggestions.md"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    try:
        analyze.main()
    finally:
        monkeypatch.undo()

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Duration p95" in content
