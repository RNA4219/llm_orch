import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import src.orch.rate_limiter as rate_limiter
from src.orch.rate_limiter import Guard


@pytest.fixture
def anyio_backend() -> str:
  return "asyncio"


@pytest.mark.anyio
async def test_guard_respects_rpm_after_wait(monkeypatch, anyio_backend: str):
  fake_time = 0.0
  sleeps: list[float] = []

  async def fake_sleep(delay: float) -> None:
    sleeps.append(delay)
    nonlocal fake_time
    fake_time += delay

  def fake_time_func() -> float:
    return fake_time

  monkeypatch.setattr(rate_limiter.time, "time", fake_time_func)
  monkeypatch.setattr(rate_limiter.asyncio, "sleep", fake_sleep)

  guard = Guard(rpm=2, concurrency=1)

  async def acquire_once() -> None:
    async with guard:
      pass

  await acquire_once()
  await acquire_once()
  await acquire_once()
  assert sleeps == [60.0]

  sleeps.clear()
  await acquire_once()
  assert sleeps == []

  await acquire_once()
  assert sleeps == [60.0]
