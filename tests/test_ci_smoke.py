import pathlib
import re

import yaml


def test_chat_curl_uses_failfast_flag() -> None:
    script_path = pathlib.Path('tools/ci/smoke.sh')
    contents = script_path.read_text().splitlines()

    curl_lines = [line for line in contents if line.strip().startswith('curl')]
    assert len(curl_lines) >= 2, 'Expected at least two curl commands in smoke.sh'

    second_curl = curl_lines[1]
    assert any(
        token.startswith('-') and 'f' in token[1:]
        for token in second_curl.split()
    ), 'Second curl invocation must include -f flag'


def test_all_curl_commands_use_failfast_flag() -> None:
    script_path = pathlib.Path('tools/ci/smoke.sh')
    contents = script_path.read_text().splitlines()

    curl_lines = [line.strip() for line in contents if line.strip().startswith('curl')]
    assert curl_lines, 'Expected at least one curl command in smoke.sh'

    for idx, curl_line in enumerate(curl_lines, start=1):
        tokens = curl_line.split()
        assert any(
            token.startswith('-') and 'f' in token.lstrip('-')
            for token in tokens[1:]
        ), f'curl command #{idx} must include the -f flag'


def test_curl_commands_support_api_key_header() -> None:
    script_path = pathlib.Path('tools/ci/smoke.sh')
    contents = script_path.read_text().splitlines()

    header_assignments = [
        line for line in contents if '-H "${ORCH_API_KEY_HEADER' in line or '-H "$ORCH_API_KEY_HEADER' in line
    ]
    assert header_assignments, 'Expected smoke.sh to define an API key header assignment'

    curl_blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in contents:
        stripped = line.strip()
        if stripped.startswith('curl'):
            current_block = [stripped]
            curl_blocks.append(current_block)
            continue
        if current_block and line.startswith(' '):
            stripped_line = line.strip()
            current_block.append(stripped_line)
            if not stripped_line.endswith('\\'):
                current_block = []
            continue
        current_block = []

    assert curl_blocks, 'Expected to find curl command blocks in smoke.sh'

    for idx, block in enumerate(curl_blocks, start=1):
        joined = ' '.join(block)
        assert '"${AUTH_HEADER_ARGS[@]}"' in joined, (
            f'curl command #{idx} must include the conditional API header arguments'
        )


def test_ci_workflow_includes_lint_jobs() -> None:
    workflow_path = pathlib.Path('.github/workflows/ci-py.yml')
    workflow_data = yaml.safe_load(workflow_path.read_text())

    jobs = workflow_data.get('jobs', {})

    assert 'ruff' in jobs, 'ci-py workflow must define a ruff lint job'
    assert 'mypy' in jobs, 'ci-py workflow must define a mypy type-check job'

    ruff_steps = jobs['ruff'].get('steps', [])
    assert any('ruff check' in step.get('run', '') for step in ruff_steps), (
        'ruff job must execute "ruff check"'
    )

    mypy_steps = jobs['mypy'].get('steps', [])
    assert any('mypy' in step.get('run', '') for step in mypy_steps), (
        'mypy job must execute mypy'
    )


def test_requirements_file_has_unique_packages() -> None:
    requirements_path = pathlib.Path('requirements.txt')
    lines = requirements_path.read_text().splitlines()

    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        requirement_part = line.split(';', 1)[0].strip()
        name_candidate = re.split(r'[<>=!~]', requirement_part, 1)[0]
        package_name = name_candidate.split('[', 1)[0].strip().lower()
        if not package_name:
            continue

        if package_name in seen and seen[package_name] != line:
            duplicates.append((package_name, seen[package_name], line))
            continue

        seen.setdefault(package_name, line)

    assert not duplicates, (
        'requirements.txt contains conflicting entries: '
        + ', '.join(
            f"{name} -> '{first}' vs '{second}'" for name, first, second in duplicates
        )
    )
