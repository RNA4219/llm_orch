from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_LEGACY_MODULE_NAME = "src.orch._providers_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "providers.py"
_LEGACY_SPEC = spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:
    raise RuntimeError("Failed to load legacy providers module.")
_LEGACY_MODULE = sys.modules.get(_LEGACY_MODULE_NAME)
if _LEGACY_MODULE is None:
    _LEGACY_MODULE = module_from_spec(_LEGACY_SPEC)
    sys.modules[_LEGACY_MODULE_NAME] = _LEGACY_MODULE
    _LEGACY_SPEC.loader.exec_module(_LEGACY_MODULE)

from .ollama import OllamaProvider

_PUBLIC_NAMES = [
    name for name in dir(_LEGACY_MODULE) if not name.startswith("_")
]
for _name in _PUBLIC_NAMES:
    globals()[_name] = getattr(_LEGACY_MODULE, _name)

globals()["OllamaProvider"] = OllamaProvider
if "OllamaProvider" not in _PUBLIC_NAMES:
    _PUBLIC_NAMES.append("OllamaProvider")

__all__ = sorted(set(_PUBLIC_NAMES))
