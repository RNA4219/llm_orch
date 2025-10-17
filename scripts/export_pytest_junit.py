from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence

_STATUS_TAGS: dict[str, str] = {
    "failure": "fail",
    "error": "error",
    "skipped": "skip",
}


def convert_junit_to_jsonl(input_path: Path, output_path: Path) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()
    records = list(_iter_testcase_records(root))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _iter_testcase_records(root: ET.Element) -> Iterable[dict[str, object]]:
    for testcase in _iter_testcases(root):
        yield _build_record(testcase)


def _iter_testcases(root: ET.Element) -> Iterable[ET.Element]:
    if root.tag == "testsuites":
        for suite in root.findall("testsuite"):
            yield from suite.findall("testcase")
        return
    if root.tag == "testsuite":
        yield from root.findall("testcase")
        return
    if root.tag == "testcase":
        yield root
        return
    for suite in root.iter("testsuite"):
        yield from suite.findall("testcase")


def _build_record(testcase: ET.Element) -> dict[str, object]:
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    time_str = testcase.attrib.get("time")
    time_value = float(time_str) if time_str else None

    record: dict[str, object] = {
        "classname": classname,
        "name": name,
        "status": "passed",
    }
    if time_value is not None:
        record["time"] = time_value

    for tag, status in _STATUS_TAGS.items():
        element = testcase.find(tag)
        if element is not None:
            record["status"] = status
            message = element.attrib.get("message")
            if message:
                record["message"] = message
            text = (element.text or "").strip()
            if text:
                record["details"] = text
            error_type = element.attrib.get("type")
            if error_type:
                record["type"] = error_type
            break

    return record


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pytest JUnit XML to JSONL")
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    convert_junit_to_jsonl(args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
