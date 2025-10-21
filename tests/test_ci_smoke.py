import pathlib
import subprocess
import sys

import pytest
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.ci.docker_build_smoke import run_docker_build_smoke


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


def test_run_docker_build_smoke_invokes_docker_build(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[object]:
        captured["command"] = command
        captured["check"] = check
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    context = pathlib.Path("/tmp/workspace")
    run_docker_build_smoke(context=context, tag="llm-orch:test")

    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[:2] == ("docker", "build")
    assert "--file" in command
    file_index = command.index("--file")
    assert command[file_index + 1] == str(context / "Dockerfile")
    assert command[-1] == str(context)
    assert captured["check"] is True


def test_ci_workflow_includes_ruff_and_mypy_steps() -> None:
    workflow_path = pathlib.Path(".github/workflows/ci-py.yml")
    assert workflow_path.exists(), "ci-py workflow file must exist"

    workflow = yaml.safe_load(workflow_path.read_text())
    assert isinstance(workflow, dict), "Workflow YAML must parse to a mapping"

    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, "Workflow must define at least one job"

    ruff_present = False
    mypy_present = False

    for job_name, job_data in jobs.items():
        if not isinstance(job_data, dict):
            continue
        steps = job_data.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            run_command = step.get("run")
            if not isinstance(run_command, str):
                continue
            if "ruff" in run_command:
                ruff_present = True
            if "mypy" in run_command:
                mypy_present = True

    assert ruff_present, "Expected at least one workflow step to run ruff"
    assert mypy_present, "Expected at least one workflow step to run mypy"
