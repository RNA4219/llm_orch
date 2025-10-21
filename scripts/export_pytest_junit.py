from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Sequence

_STATUS_TAGS: dict[str, str] = {
    "failure": "fail",
    "error": "fail",
    "skipped": "skip",
}


def _strip_namespace(tag: str) -> str:
    """Return the local tag name without any XML namespace prefix."""
    if "}" not in tag:
        return tag
    return tag.split("}", 1)[1]


def convert_junit_to_jsonl(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return

    tree = ET.parse(input_path)
    root = tree.getroot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in _iter_testcase_records(root):
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _iter_testcase_records(root: ET.Element) -> Iterable[dict[str, object]]:
    for testcase in _iter_testcases(root):
        yield _build_record(testcase)


def _parse_duration_ms(time_str: str | None) -> int | None:
    if not time_str:
        return None
    try:
        milliseconds = Decimal(time_str) * Decimal(1000)
        quantized = milliseconds.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return int(quantized)
    except (InvalidOperation, OverflowError, ValueError):
        return None


def _iter_testcases(root: ET.Element) -> Iterable[ET.Element]:
    tag = _strip_namespace(root.tag)

    if tag == "testcase":
        yield root
        return

    for child in root:
        yield from _iter_testcases(child)


def _build_record(testcase: ET.Element) -> dict[str, object]:
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    duration_ms = _parse_duration_ms(testcase.attrib.get("time"))

    record: dict[str, object] = {
        "classname": classname,
        "name": name,
        "status": "passed",
    }
    if duration_ms is not None:
        record["duration_ms"] = duration_ms

    for child in testcase:
        tag = _strip_namespace(child.tag)
        status = _STATUS_TAGS.get(tag)
        if status is None:
            continue

        record["status"] = status
        message = child.attrib.get("message")
        if message:
            record["message"] = message
        text = (child.text or "").strip()
        if text:
            record["details"] = text
        error_type = child.attrib.get("type")
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
