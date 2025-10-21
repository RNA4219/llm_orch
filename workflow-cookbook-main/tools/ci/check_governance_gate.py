# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 RNA4219

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Sequence


def _strip_inline_comment(value: str) -> str:
    comment_stripped: list[str] = []
    in_single_quote = False
    in_double_quote = False
    for character in value:
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif character == "#" and not in_single_quote and not in_double_quote:
            break
        comment_stripped.append(character)
    return "".join(comment_stripped).strip()


def load_forbidden_patterns(policy_path: Path) -> List[str]:
    patterns: List[str] = []
    in_self_modification = False
    in_forbidden_paths = False
    forbidden_indent: int | None = None

    for raw_line in policy_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))

        if stripped_line.endswith(":"):
            key = stripped_line[:-1].strip()
            if indent == 0:
                in_self_modification = key == "self_modification"
                in_forbidden_paths = False
                forbidden_indent = None
            elif in_self_modification and key == "forbidden_paths":
                in_forbidden_paths = True
                forbidden_indent = indent
            elif indent <= (forbidden_indent or indent):
                in_forbidden_paths = False
            continue

        if in_forbidden_paths and stripped_line.startswith("- "):
            value = stripped_line[2:].strip()
            value = _strip_inline_comment(value)
            if len(value) >= 2 and value[0] in {'"', "'"} and value[-1] == value[0]:
                value = value[1:-1]
            if value:
                patterns.append(value.lstrip("/"))
            continue

        if in_forbidden_paths and indent <= (forbidden_indent or indent):
            in_forbidden_paths = False

    return patterns


def get_changed_paths(refspec: str) -> List[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", refspec],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _normalize_forbidden_value(value: str) -> str:
    trimmed = value.lstrip("/")
    while trimmed.startswith("./") or trimmed.startswith("../"):
        if trimmed.startswith("./"):
            trimmed = trimmed[2:]
        else:
            trimmed = trimmed[3:]
        trimmed = trimmed.lstrip("/")
    if not trimmed:
        return ""
    return PurePosixPath(trimmed).as_posix()


def find_forbidden_matches(paths: Iterable[str], patterns: Sequence[str]) -> List[str]:
    matches: List[str] = []
    for path in paths:
        normalized_path = _normalize_forbidden_value(path)
        path_object = PurePosixPath(normalized_path)
        for pattern in patterns:
            normalized_pattern = _normalize_forbidden_value(pattern)
            if not normalized_pattern:
                if not normalized_path:
                    matches.append(normalized_path)
                    break
                continue
            if path_object.match(normalized_pattern):
                matches.append(normalized_path)
                break
            if normalized_pattern.endswith("/**"):
                prefix = normalized_pattern[:-3]
                if not prefix:
                    matches.append(normalized_path)
                    break
                try:
                    if path_object.is_relative_to(prefix):
                        matches.append(normalized_path)
                        break
                except ValueError:
                    continue
    return matches


def read_event_body(event_path: Path) -> str | None:
    if not event_path.exists():
        return None
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return None
    body = pull_request.get("body")
    if body is None:
        return None
    if not isinstance(body, str):
        return None
    return body


def validate_priority_score(body: str | None) -> bool:
    if body is None:
        return False

    header = "Priority Score:"
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith(header):
            continue

        remainder = line[len(header) :].strip()
        if not remainder or remainder.startswith("<!--"):
            continue

        if "/" not in remainder:
            continue

        score_text, reason = (part.strip() for part in remainder.split("/", 1))
        if not score_text.isdigit():
            continue

        score = int(score_text)
        if score < 1 or score > 5:
            continue

        if not reason or reason.startswith("<!--"):
            continue

        return True

    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    policy_path = repo_root / "governance" / "policy.yaml"
    forbidden_patterns = load_forbidden_patterns(policy_path)

    try:
        changed_paths = get_changed_paths("origin/main...")
    except subprocess.CalledProcessError as error:
        print(f"Failed to collect changed paths: {error}", file=sys.stderr)
        return 1
    violations = find_forbidden_matches(changed_paths, forbidden_patterns)
    if violations:
        print(
            "Forbidden path modifications detected:\n" + "\n".join(f" - {path}" for path in violations),
            file=sys.stderr,
        )
        return 1

    event_path_value = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path_value:
        print("GITHUB_EVENT_PATH is not set", file=sys.stderr)
        return 1
    body = read_event_body(Path(event_path_value))
    if not validate_priority_score(body):
        print("Priority score validation failed", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
