from __future__ import annotations

from pathlib import Path

import yaml


def test_reflection_download_artifact_allows_missing() -> None:
    workflow_path = Path(".github/workflows/reflection.yml")
    content = yaml.safe_load(workflow_path.read_text())
    steps = content["jobs"]["reflect"]["steps"]
    download_step = next(
        step for step in steps if step.get("uses") == "actions/download-artifact@v4"
    )
    assert download_step["with"]["if-no-artifact-found"] == "ignore"
