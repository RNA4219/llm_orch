import pathlib
import re


def _read_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_all_smoke_curl_commands_use_failfast_flag() -> None:
    script_path = pathlib.Path("tools/ci/smoke.sh")
    contents = _read_lines(script_path)

    curl_lines = [line.strip() for line in contents if line.strip().startswith("curl")]
    assert len(curl_lines) >= 2, "Expected at least two curl commands in smoke.sh"

    first_curl_tokens = curl_lines[0].split()
    assert "-f" in first_curl_tokens, "Health check curl must include -f flag"

    second_curl_tokens = curl_lines[1].split()
    assert "-fs" in second_curl_tokens, "Chat curl must include -fs flags"


def test_changelog_file_exists_with_semver_tags() -> None:
    changelog_path = pathlib.Path("CHANGELOG.md")
    assert changelog_path.exists(), "CHANGELOG.md must exist at repository root"

    version_heading_pattern = re.compile(r"^##\s+v\d+\.\d+\.\d+(\s+-\s+\d{4}-\d{2}-\d{2})?$")
    headings = [line.strip() for line in _read_lines(changelog_path) if line.startswith("## ")]
    assert headings, "CHANGELOG.md must contain at least one version heading"

    invalid_headings = [heading for heading in headings if version_heading_pattern.match(heading) is None]
    assert not invalid_headings, "All version headings must use format '## vMAJOR.MINOR.PATCH - YYYY-MM-DD'"


def test_version_file_uses_semver() -> None:
    version_path = pathlib.Path("VERSION")
    assert version_path.exists(), "VERSION file must exist at repository root"

    version = version_path.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), "VERSION must contain MAJOR.MINOR.PATCH"
