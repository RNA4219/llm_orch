from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

_IMPL_ALIAS = "src.orch._providers_impl"


def _load_impl() -> ModuleType:
    existing = sys.modules.get(_IMPL_ALIAS)
    if existing is not None:
        return existing
    impl_path = pathlib.Path(__file__).resolve().parents[1] / "providers.py"
    spec = importlib.util.spec_from_file_location(_IMPL_ALIAS, impl_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to load providers implementation from {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL_ALIAS] = module
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

BaseProvider = _impl.BaseProvider
OpenAICompatProvider = _impl.OpenAICompatProvider
AnthropicProvider = _impl.AnthropicProvider
OllamaProvider = _impl.OllamaProvider
DummyProvider = _impl.DummyProvider
ProviderRegistry = _impl.ProviderRegistry
UnsupportedContentBlockError = _impl.UnsupportedContentBlockError
_normalize_anthropic_tool = _impl._normalize_anthropic_tool
_normalize_anthropic_tools = _impl._normalize_anthropic_tools
_normalize_anthropic_tool_choice = _impl._normalize_anthropic_tool_choice

__all__ = [
    "BaseProvider",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "DummyProvider",
    "ProviderRegistry",
    "UnsupportedContentBlockError",
    "_normalize_anthropic_tool",
    "_normalize_anthropic_tools",
    "_normalize_anthropic_tool_choice",
]
