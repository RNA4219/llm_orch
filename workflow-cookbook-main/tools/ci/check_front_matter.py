from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REQUIRED_FIELDS: Sequence[str] = (
    "intent_id",
    "owner",
    "status",
    "last_reviewed_at",
    "next_review_due",
)

INCIDENT_REQUIRED_FIELDS: Sequence[str] = (
    "incident_id",
    "occurred_at",
    "owner",
    "status",
    "linked_pr",
    "runbook",
)


def _extract_front_matter_lines(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[:1] != ["---"]:
        return []
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration:
        return []
    return lines[1:end]


def _parse_fields(front_matter_lines: Iterable[str]) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw in front_matter_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        for index, char in enumerate(value):
            if char == "#" and (index == 0 or value[index - 1].isspace()):
                value = value[:index].rstrip()
                break
        data[key.strip()] = value
    return data


def validate_markdown_front_matter(root: Path) -> Dict[Path, List[str]]:
    missing: Dict[Path, List[str]] = {}
    for md_path in sorted(root.glob("*.md")):
        fields = _parse_fields(_extract_front_matter_lines(md_path))
        absent = [field for field in REQUIRED_FIELDS if not fields.get(field)]
        if absent:
            missing[md_path] = absent if fields else list(REQUIRED_FIELDS)
    return missing


def validate_incident_front_matter(root: Path) -> Dict[Path, List[str]]:
    missing: Dict[Path, List[str]] = {}
    docs_dir = root / "docs"
    if not docs_dir.is_dir():
        return missing
    for md_path in sorted(docs_dir.glob("IN-*.md")):
        fields = _parse_fields(_extract_front_matter_lines(md_path))
        absent = [field for field in INCIDENT_REQUIRED_FIELDS if not fields.get(field)]
        if absent:
            missing[md_path] = absent if fields else list(INCIDENT_REQUIRED_FIELDS)
    return missing


def _format_missing(missing: Dict[Path, List[str]]) -> str:
    return "\n".join(f"{path}: missing {', '.join(fields)}" for path, fields in missing.items())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate front matter in top-level Markdown files.")
    parser.add_argument("--check", action="store_true", help="Validate and exit with non-zero status on missing fields.")
    parser.add_argument("root", nargs="?", default=Path.cwd(), type=Path, help="Repository root to scan (default: cwd).")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    missing: Dict[Path, List[str]] = {}
    missing.update(validate_markdown_front_matter(root))
    missing.update(validate_incident_front_matter(root))
    if not missing:
        return 0
    message = _format_missing(missing)
    if args.check:
        print(message, file=sys.stderr)
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
