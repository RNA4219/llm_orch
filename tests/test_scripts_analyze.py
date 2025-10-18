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


def test_load_results_counts_error_status_as_failure(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    log_path.parent.mkdir(parents=True)

    records = [
        {"name": "sample::error_case", "duration_ms": 10, "status": "error"},
    ]

    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)

    _, _, fails = analyze.load_results()

    assert fails == ["sample::error_case"]


def test_load_results_counts_multiple_failure_statuses(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    log_path.parent.mkdir(parents=True)

    records = [
        {"name": "sample::error_case", "duration_ms": 5, "status": "error"},
        {"name": "sample::failed_case", "duration_ms": 7, "status": "failed"},
        {"name": "sample::errored_case", "duration_ms": 9, "status": "errored"},
        {"name": "sample::pass_case", "duration_ms": 11, "status": "pass"},
    ]

    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)

    _, _, fails = analyze.load_results()

    assert fails == [
        "sample::error_case",
        "sample::failed_case",
        "sample::errored_case",
    ]


def test_load_results_ignores_missing_duration_records(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    log_path.parent.mkdir(parents=True)

    records = [
        {"name": "sample::no_duration", "status": "pass"},
        {"name": "sample::none_duration", "duration_ms": None, "status": "pass"},
        {"name": "sample::with_duration", "duration_ms": 15, "status": "pass"},
    ]

    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)

    _, durations, _ = analyze.load_results()

    assert durations == [15]


def test_load_results_handles_utf8_content(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    log_path.parent.mkdir(parents=True)

    records = [
        {"name": "サンプル::成功", "duration_ms": 12, "status": "pass"},
        {"name": "サンプル::失敗", "duration_ms": 34, "status": "fail"},
    ]

    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)

    tests, _, fails = analyze.load_results()

    assert tests == ["サンプル::成功", "サンプル::失敗"]
    assert fails == ["サンプル::失敗"]


def test_analyze_main_handles_blank_lines(tmp_path, monkeypatch):
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
        fp.write("\n")
        for record in records:
            fp.write(json.dumps(record) + "\n\n")
        fp.write("   \n")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    assert report_path.exists(), "Report file should be generated even with blank lines"


def test_analyze_handles_invalid_json_line(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps({"name": "sample::good_one", "duration_ms": 10, "status": "pass"}) + "\n")
        fp.write("{bad json" + "\n")
        fp.write(json.dumps({"name": "sample::good_two", "duration_ms": 30, "status": "fail"}) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    tests, _, fails = analyze.load_results()

    assert tests == ["sample::good_one", "sample::good_two"]
    assert fails == ["sample::good_two"]

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Total tests: 2" in report_text


def test_analyze_main_skip_only_counts_as_not_run(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    record = {"name": "sample::skipped", "duration_ms": 42, "status": "skip"}
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Total tests: 0" in report_text
    assert "- Pass rate: 未実行" in report_text


def test_analyze_main_reports_no_tests_when_log_missing(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    report_path.parent.mkdir(parents=True)

    issue_path.write_text("stale", encoding="utf-8")

    monkeypatch.setattr(analyze, "LOG", tmp_path / "logs" / "missing.jsonl")
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Total tests: 0" in report_text
    assert "- Pass rate: 未実行" in report_text

    if issue_path.exists():
        assert issue_path.read_text(encoding="utf-8") == ""
    else:
        assert not issue_path.exists()


def test_analyze_main_clears_issue_suggestions_when_no_failures(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    log_records = [
        {"name": "sample::test_one", "duration_ms": 15, "status": "pass"},
        {"name": "sample::test_two", "duration_ms": 30, "status": "pass"},
    ]
    with log_path.open("w", encoding="utf-8") as fp:
        for record in log_records:
            fp.write(json.dumps(record) + "\n")

    issue_path.write_text("outdated", encoding="utf-8")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    if issue_path.exists():
        assert issue_path.read_text(encoding="utf-8") == ""
    else:
        assert not issue_path.exists()


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


def test_analyze_main_treats_skipped_tests_as_unexecuted(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    records = [
        {"name": "sample::test_skip_one", "duration_ms": 5, "status": "skip"},
        {"name": "sample::test_skip_two", "duration_ms": 10, "status": "skipped"},
    ]

    with log_path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Total tests: 0" in report_text
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


def test_analyze_main_handles_null_duration(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    record = {"name": "sample::null", "duration_ms": None, "status": "pass"}
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    report_text = report_path.read_text(encoding="utf-8")
    assert "- Duration p95: 0 ms" in report_text


def test_analyze_main_removes_stale_issue_suggestions(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "test.jsonl"
    report_path = tmp_path / "reports" / "today.md"
    issue_path = tmp_path / "reports" / "issue_suggestions.md"

    log_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    record = {"name": "sample::solo", "duration_ms": 5, "status": "pass"}
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")

    issue_path.write_text("stale content", encoding="utf-8")

    monkeypatch.setattr(analyze, "LOG", log_path)
    monkeypatch.setattr(analyze, "REPORT", report_path)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue_path)

    analyze.main()

    if issue_path.exists():
        assert issue_path.read_text(encoding="utf-8") == ""
    else:
        assert not issue_path.exists()
