import json
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.export_pytest_junit import convert_junit_to_jsonl


def write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def read_json_lines(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


@pytest.mark.parametrize(
    "xml_content",
    [
        (
            """
            <testsuite name="sample" tests="2" failures="1" errors="0" skipped="0">
                <testcase classname="sample.TestCase" name="test_success" time="0.123" />
                <testcase classname="sample.TestCase" name="test_failure" time="0.456">
                    <failure message="assertion failed">AssertionError</failure>
                </testcase>
            </testsuite>
            """
        ),
    ],
)
def test_convert_junit_to_jsonl_records_passed_and_failed(tmp_path: Path, xml_content: str) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(xml_path, xml_content)

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "sample.TestCase",
            "name": "test_success",
            "status": "passed",
            "time": 0.123,
        },
        {
            "classname": "sample.TestCase",
            "details": "AssertionError",
            "message": "assertion failed",
            "name": "test_failure",
            "status": "fail",
            "time": 0.456,
        },
    ]


def test_convert_junit_to_jsonl_handles_skipped_and_errors(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuites>
            <testsuite name="suite_one" tests="3" failures="0" errors="1" skipped="1">
                <testcase classname="a.TestCase" name="test_error" time="0.001">
                    <error type="RuntimeError" message="boom">Traceback</error>
                </testcase>
                <testcase classname="a.TestCase" name="test_skipped" time="0.0">
                    <skipped message="not supported">reason</skipped>
                </testcase>
                <testcase classname="a.TestCase" name="test_pass" time="0.01" />
            </testsuite>
        </testsuites>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "a.TestCase",
            "details": "Traceback",
            "message": "boom",
            "name": "test_error",
            "status": "error",
            "time": 0.001,
            "type": "RuntimeError",
        },
        {
            "classname": "a.TestCase",
            "details": "reason",
            "message": "not supported",
            "name": "test_skipped",
            "status": "skip",
            "time": 0.0,
        },
        {
            "classname": "a.TestCase",
            "name": "test_pass",
            "status": "passed",
            "time": 0.01,
        },
    ]


def test_convert_junit_to_jsonl_normalizes_status_values(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite name="sample" tests="2" failures="1" errors="0" skipped="1">
            <testcase classname="sample.TestCase" name="test_failure">
                <failure message="oops" />
            </testcase>
            <testcase classname="sample.TestCase" name="test_skipped">
                <skipped message="nope" />
            </testcase>
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert [record["status"] for record in records] == ["fail", "skip"]
