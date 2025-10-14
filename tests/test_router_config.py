import importlib
import sys
from pathlib import Path

import pytest


def _load_router_module():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    importlib.invalidate_caches()
    return importlib.import_module("src.orch.router")


def test_load_config_raises_when_fallback_provider_missing(tmp_path):
    load_config = _load_router_module().load_config
    config_dir = tmp_path
    (config_dir / "providers.toml").write_text(
        """[existing]\nbase_url = \"https://example.com\"\nmodel = \"gpt\"\n""",
        encoding="utf-8",
    )
    (config_dir / "router.yaml").write_text(
        """routes:\n  DEFAULT:\n    primary: existing\n    fallback:\n      - missing\n""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fallback"):
        load_config(str(config_dir))
