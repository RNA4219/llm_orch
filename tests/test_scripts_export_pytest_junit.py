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


def test_convert_junit_to_jsonl_normalizes_failed_status(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite>
            <testcase classname="pkg.TestCase" name="test_case" time="0.1">
                <failure message="boom">AssertionError</failure>
            </testcase>
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "details": "AssertionError",
            "message": "boom",
            "name": "test_case",
            "status": "fail",
            "duration_ms": 100,
        }
    ]


def test_convert_junit_to_jsonl_includes_duration_ms(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite>
            <testcase classname="pkg.TestCase" name="test_case" time="0.123" />
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "duration_ms": 123,
            "name": "test_case",
            "status": "passed",
        }
    ]
    assert "time" not in records[0]


def test_convert_junit_to_jsonl_rounds_duration_ms(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite>
            <testcase classname="pkg.TestCase" name="test_case" time="0.0015" />
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "duration_ms": 2,
            "name": "test_case",
            "status": "passed",
        }
    ]


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
            "duration_ms": 123,
        },
        {
            "classname": "sample.TestCase",
            "details": "AssertionError",
            "message": "assertion failed",
            "name": "test_failure",
            "status": "fail",
            "duration_ms": 456,
        },
    ]


def test_convert_junit_to_jsonl_sets_fail_status_for_failure(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testcase classname="pkg.TestCase" name="test_failure">
            <failure />
        </testcase>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "name": "test_failure",
            "status": "fail",
        }
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
            "duration_ms": 1,
            "type": "RuntimeError",
        },
        {
            "classname": "a.TestCase",
            "details": "reason",
            "message": "not supported",
            "name": "test_skipped",
            "status": "skip",
            "duration_ms": 0,
        },
        {
            "classname": "a.TestCase",
            "name": "test_pass",
            "status": "passed",
            "duration_ms": 10,
        },
    ]
