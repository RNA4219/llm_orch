from pathlib import Path

import yaml


def test_pr_gate_uses_merge_info_preview_header() -> None:
    workflow_path = Path('.github/workflows/pr_gate.yml')
    workflow = yaml.safe_load(workflow_path.read_text())
    steps = workflow['jobs']['gate']['steps']
    check_step = next(step for step in steps if step.get('name') == 'Check CODEOWNERS approval')
    script = check_step['with']['script']
    assert 'application/vnd.github.merge-info-preview+json' in script


def test_pr_gate_normalizes_unknown_decision_to_review_required() -> None:
    workflow_path = Path('.github/workflows/pr_gate.yml')
    workflow = yaml.safe_load(workflow_path.read_text())
    steps = workflow['jobs']['gate']['steps']
    check_step = next(step for step in steps if step.get('name') == 'Check CODEOWNERS approval')
    script = check_step['with']['script']
    normalized_script = script.replace(' ', '').replace('\n', '')
    assert (
        "decisionRaw==='UNKNOWN'||decisionRaw===null||decisionRaw===undefined?'REVIEW_REQUIRED'"
        in normalized_script
    )
