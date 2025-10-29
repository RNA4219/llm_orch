import pathlib
from typing import Sequence, Tuple

import scripts.analyze as analyze


def _run_main(
    monkeypatch,
    tmp_path: pathlib.Path,
    results: Tuple[Sequence[str], Sequence[int], Sequence[str]],
) -> pathlib.Path:
    report = tmp_path / "reports" / "today.md"
    issue = tmp_path / "reports" / "issue_suggestions.md"
    log = tmp_path / "logs" / "test.jsonl"
    monkeypatch.setattr(analyze, "REPORT", report)
    monkeypatch.setattr(analyze, "ISSUE_OUT", issue)
    monkeypatch.setattr(analyze, "LOG", log)
    monkeypatch.setattr(analyze, "load_results", lambda: results)
    analyze.main()
    return issue


def test_main_writes_issue_list_when_failures(monkeypatch, tmp_path):
    issue_path = _run_main(
        monkeypatch,
        tmp_path,
        (["t1"], [100], ["t1", "t1"]),
    )
    content = issue_path.read_text(encoding="utf-8")
    assert "反省TODO" in content
    assert "t1" in content


def test_main_removes_issue_file_when_no_failures(monkeypatch, tmp_path):
    issue_path = tmp_path / "reports" / "issue_suggestions.md"
    issue_path.parent.mkdir(parents=True, exist_ok=True)
    issue_path.write_text("dummy", encoding="utf-8")
    _run_main(monkeypatch, tmp_path, ([], [], []))
    assert not issue_path.exists()
