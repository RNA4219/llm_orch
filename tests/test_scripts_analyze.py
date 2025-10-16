import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.analyze as analyze


def test_analyze_main_generates_report(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    records = [
        {"name": "sample::test_one", "duration_ms": 10, "status": "pass"},
        {"name": "sample::test_two", "duration_ms": 20, "status": "fail"},
    ]
    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    assert report_path.exists(), "Report file should be generated"


def test_analyze_main_reports_no_tests_when_log_missing(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    report_path.parent.mkdir(parents=True)

    monkeypatch.setattr(analyze, "LOG", tmp_path / "logs" / "missing.jsonl")
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Total tests: 0" in report_text
    assert "- Pass rate: 未実行" in report_text


def test_analyze_main_handles_empty_log(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "empty.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)
    log_path.touch()

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Pass rate: 未実行" in report_text


def test_analyze_main_single_record_p95(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    record = {"name": "sample::solo", "duration_ms": 123, "status": "pass"}
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    assert "- Duration p95: 123 ms" in report_path.read_text(encoding="utf-8")
