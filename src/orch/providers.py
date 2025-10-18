"""Compatibility facade for :mod:`src.orch.providers`."""

# [ ] anthropic移行完了

from . import providers as _providers
from .providers import *  # noqa: F401,F403

__all__ = _providers.__all__
