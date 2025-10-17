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


def test_convert_junit_to_jsonl_records_duration_ms(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite>
            <testcase classname="sample.TestCase" name="test_case" time="0.250" />
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    [record] = read_json_lines(output_path)
    assert record["duration_ms"] == 250
    assert "time" not in record


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

    [record] = read_json_lines(output_path)
    assert record["duration_ms"] == 2


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
            "status": "fail",
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


def test_convert_junit_to_jsonl_handles_nested_testsuites(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite name="root">
            <testsuite name="child_one">
                <testcase classname="pkg.TestCase" name="test_first" time="0.1" />
            </testsuite>
            <testsuite name="child_two">
                <testcase classname="pkg.TestCase" name="test_second" time="0.2" />
            </testsuite>
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "name": "test_first",
            "status": "passed",
            "duration_ms": 100,
        },
        {
            "classname": "pkg.TestCase",
            "name": "test_second",
            "status": "passed",
            "duration_ms": 200,
        },
    ]


def test_convert_junit_to_jsonl_handles_default_namespace(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite xmlns="urn:pytest">
            <testcase classname="pkg.TestCase" name="test_one" time="0.100" />
            <testcase classname="pkg.TestCase" name="test_two" time="0.200" />
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "duration_ms": 100,
            "name": "test_one",
            "status": "passed",
        },
        {
            "classname": "pkg.TestCase",
            "duration_ms": 200,
            "name": "test_two",
            "status": "passed",
        },
    ]


def test_convert_junit_to_jsonl_normalizes_error_status(tmp_path: Path) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuite>
            <testcase classname="pkg.TestCase" name="test_error">
                <error message="boom">Traceback</error>
            </testcase>
        </testsuite>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "details": "Traceback",
            "message": "boom",
            "name": "test_error",
            "status": "fail",
        }
    ]


def test_convert_junit_to_jsonl_handles_nested_testsuites_within_testsuites(
    tmp_path: Path,
) -> None:
    xml_path = tmp_path / "pytest.xml"
    output_path = tmp_path / "out.jsonl"
    write_file(
        xml_path,
        """
        <testsuites name="root">
            <testsuite name="parent">
                <testsuite name="child">
                    <testcase classname="pkg.TestCase" name="test_inner" time="0.123" />
                </testsuite>
                <testcase classname="pkg.TestCase" name="test_sibling" time="0.456" />
            </testsuite>
        </testsuites>
        """,
    )

    convert_junit_to_jsonl(xml_path, output_path)

    records = read_json_lines(output_path)
    assert records == [
        {
            "classname": "pkg.TestCase",
            "name": "test_inner",
            "status": "passed",
            "duration_ms": 123,
        },
        {
            "classname": "pkg.TestCase",
            "name": "test_sibling",
            "status": "passed",
            "duration_ms": 456,
        },
    ]
