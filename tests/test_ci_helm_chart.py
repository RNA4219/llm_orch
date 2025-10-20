import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("helm") is None, reason="helm CLI is not available")
def test_helm_template_injects_configs(tmp_path: Path) -> None:
    chart_dir = Path(__file__).resolve().parent.parent / "charts" / "llm-orch"
    assert chart_dir.exists(), "Helm chart directory is missing"

    values_file = tmp_path / "values.yaml"
    values_file.write_text(
        """
config:
  providers: |
    [providers.test]
    model = "gpt-4"
  router: |
    routes:
      - name: default
        primary: providers.test
        fallback: []
""".strip()
    )

    result = subprocess.run(
        [
            "helm",
            "template",
            "llm-orch",
            str(chart_dir),
            "-f",
            str(values_file),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    sanitized_lines = [
        line
        for line in result.stdout.splitlines()
        if not line.startswith("# Source:")
    ]
    rendered = "\n".join(sanitized_lines)

    assert "providers.toml: |" in rendered
    assert "router.yaml: |" in rendered
    assert "model = \"gpt-4\"" in rendered
    assert "primary: providers.test" in rendered
