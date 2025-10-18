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
async def test_guard_does_not_exceed_single_rpm(
  monkeypatch: pytest.MonkeyPatch, anyio_backend: str
) -> None:
  _ = anyio_backend
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

  guard = Guard(rpm=1, concurrency=1)

  async def acquire_once() -> None:
    async with guard:
      pass

  await acquire_once()
  assert sleeps == []

  await acquire_once()
  assert sleeps == [60.0]

  await acquire_once()
  assert sleeps == [60.0, 60.0]


@pytest.mark.anyio
async def test_guard_tpm_sliding_window_behavior(
  monkeypatch: pytest.MonkeyPatch, anyio_backend: str
) -> None:
  _ = anyio_backend
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

  guard = Guard(rpm=100, concurrency=1, tpm=100)

  lease1: rate_limiter.GuardLease | None = None
  async with guard.acquire(estimated_prompt_tokens=40) as lease:
    lease1 = lease
  assert lease1 is not None
  wait = guard.record_usage(
    lease1,
    usage_prompt_tokens=40,
    usage_completion_tokens=0,
  )
  assert wait == pytest.approx(0.0)
  assert sleeps == []

  lease2: rate_limiter.GuardLease | None = None
  async with guard.acquire(estimated_prompt_tokens=70) as lease:
    lease2 = lease
  assert lease2 is not None
  assert sleeps == [60.0]
  wait = guard.record_usage(
    lease2,
    usage_prompt_tokens=50,
    usage_completion_tokens=10,
  )
  assert wait == pytest.approx(0.0)

  lease3: rate_limiter.GuardLease | None = None
  async with guard.acquire(estimated_prompt_tokens=10) as lease:
    lease3 = lease
  assert lease3 is not None
  wait = guard.record_usage(
    lease3,
    usage_prompt_tokens=20,
    usage_completion_tokens=40,
  )
  assert wait == pytest.approx(60.0)
  assert sleeps == [60.0]

  fake_time = 120.0

  lease4: rate_limiter.GuardLease | None = None
  async with guard.acquire(estimated_prompt_tokens=30) as lease:
    lease4 = lease
  assert lease4 is not None
  wait = guard.record_usage(
    lease4,
    usage_prompt_tokens=0,
    usage_completion_tokens=0,
  )
  assert wait == pytest.approx(0.0)
  assert sleeps == [60.0]

  lease5: rate_limiter.GuardLease | None = None
  async with guard.acquire(estimated_prompt_tokens=30) as lease:
    lease5 = lease
  assert lease5 is not None
  wait = guard.record_usage(
    lease5,
    usage_prompt_tokens=30,
    usage_completion_tokens=0,
  )
  assert wait == pytest.approx(0.0)
  assert sleeps == [60.0]


@pytest.mark.anyio
async def test_guard_record_usage_wait_time(monkeypatch: pytest.MonkeyPatch) -> None:
  fake_time = 0.0

  async def fake_sleep(delay: float) -> None:
    nonlocal fake_time
    fake_time += delay

  def fake_time_func() -> float:
    return fake_time

  monkeypatch.setattr(rate_limiter.time, "time", fake_time_func)
  monkeypatch.setattr(rate_limiter.asyncio, "sleep", fake_sleep)

  guard = Guard(rpm=100, concurrency=1, tpm=100)

  lease: rate_limiter.GuardLease | None = None
  async with guard.acquire(estimated_prompt_tokens=50) as acquired:
    lease = acquired
  assert lease is not None
  wait = guard.record_usage(
    lease,
    usage_prompt_tokens=80,
    usage_completion_tokens=60,
  )
  assert wait == pytest.approx(60.0)
