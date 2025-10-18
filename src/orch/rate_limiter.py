import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List

from .router import ProviderDef


class TokenBucket:
  def __init__(self, rpm: int):
    self.capacity = max(1, rpm)
    self.tokens = self.capacity
    self.window_start = int(time.time() // 60)

  def try_take(self) -> float:
    now_min = int(time.time() // 60)
    if now_min != self.window_start:
      self.window_start = now_min
      self.tokens = self.capacity
    if self.tokens > 0:
      self.tokens -= 1
      return 0.0
    now = time.time()
    return 60 - (now % 60)


@dataclass
class GuardLease:
  reservation_id: int | None
  estimated_prompt_tokens: int
  acquired_at: float
  committed: bool = False


@dataclass
class _WindowEntry:
  entry_id: int
  timestamp: float
  tokens: int


class SlidingWindowBucket:
  def __init__(self, tpm: int, window_seconds: float = 60.0):
    self.capacity = max(1, tpm)
    self.window_seconds = window_seconds
    self._entries: Deque[_WindowEntry] = deque()
    self._index: dict[int, _WindowEntry] = {}
    self._total = 0
    self._next_id = 1

  def _prune(self, now: float) -> None:
    cutoff = now - self.window_seconds
    while self._entries:
      entry = self._entries[0]
      if entry.tokens <= 0 or entry.timestamp <= cutoff:
        if entry.tokens > 0:
          self._total -= entry.tokens
        self._entries.popleft()
        self._index.pop(entry.entry_id, None)
        continue
      break

  def _time_until_reduction(self, needed: int, now: float) -> float:
    remaining = needed
    wait = 0.0
    for entry in self._entries:
      if entry.tokens <= 0:
        continue
      remaining -= entry.tokens
      candidate = entry.timestamp + self.window_seconds - now
      if candidate > wait:
        wait = candidate
      if remaining <= 0:
        break
    return max(wait, 0.0)

  def reserve(self, tokens: int, now: float) -> tuple[float, int | None]:
    self._prune(now)
    if tokens <= 0:
      return 0.0, None
    if self._total + tokens <= self.capacity:
      entry_id = self._next_id
      self._next_id += 1
      entry = _WindowEntry(entry_id, now, tokens)
      self._entries.append(entry)
      self._index[entry_id] = entry
      self._total += tokens
      return 0.0, entry_id
    needed = self._total + tokens - self.capacity
    wait = self._time_until_reduction(needed, now)
    return wait, None

  def cancel(self, reservation_id: int | None, now: float) -> None:
    if reservation_id is None:
      return
    self._prune(now)
    entry = self._index.get(reservation_id)
    if entry is None:
      return
    if entry.tokens > 0:
      self._total -= entry.tokens
    entry.tokens = 0
    entry.timestamp = now
    self._prune(now)

  def commit(self, reservation_id: int | None, tokens: int, now: float) -> float:
    self._prune(now)
    actual = max(0, tokens)
    entry = self._index.get(reservation_id) if reservation_id is not None else None
    if entry is None:
      if actual == 0:
        return 0.0
      entry_id = self._next_id
      self._next_id += 1
      entry = _WindowEntry(entry_id, now, actual)
      self._entries.append(entry)
      self._index[entry_id] = entry
      self._total += actual
    else:
      if entry.tokens > 0:
        self._total -= entry.tokens
      entry.tokens = actual
      entry.timestamp = now
      if entry.tokens > 0:
        self._total += entry.tokens
      else:
        self._index.pop(entry.entry_id, None)
    self._prune(now)
    if self._total <= self.capacity:
      return 0.0
    excess = self._total - self.capacity
    return self._time_until_reduction(excess, now)


class Guard:
  def __init__(self, rpm: int, concurrency: int, tpm: int | None = None):
    self.bucket = TokenBucket(rpm)
    self.sem = asyncio.Semaphore(concurrency)
    self._tpm_bucket = SlidingWindowBucket(tpm) if tpm is not None else None
    self._leases: Dict[asyncio.Task[object], List[GuardLease]] = {}

  def acquire(self, estimated_prompt_tokens: int = 0) -> "_GuardContext":
    return _GuardContext(self, estimated_prompt_tokens)

  __call__ = acquire

  async def __aenter__(self) -> GuardLease:
    return await self._acquire(0)

  async def __aexit__(self, exc_type, exc, tb):
    self._release_current_task()

  async def _acquire(self, estimated_prompt_tokens: int) -> GuardLease:
    estimate = max(0, estimated_prompt_tokens)
    while True:
      now = time.time()
      reservation_id: int | None = None
      if self._tpm_bucket is not None:
        delay_tokens, reservation_id = self._tpm_bucket.reserve(estimate, now)
        if delay_tokens > 0:
          await asyncio.sleep(delay_tokens)
          continue
      delay_rpm = self.bucket.try_take()
      if delay_rpm > 0:
        if self._tpm_bucket is not None and reservation_id is not None:
          self._tpm_bucket.cancel(reservation_id, now)
        await asyncio.sleep(delay_rpm)
        continue
      await self.sem.acquire()
      lease = GuardLease(reservation_id=reservation_id, estimated_prompt_tokens=estimate, acquired_at=now)
      self._register_current_task(lease)
      return lease

  def _register_current_task(self, lease: GuardLease) -> None:
    task = asyncio.current_task()
    if task is None:
      raise RuntimeError("guard context requires running task")
    stack = self._leases.setdefault(task, [])
    stack.append(lease)

  def _release_current_task(self) -> None:
    task = asyncio.current_task()
    if task is None:
      raise RuntimeError("guard context requires running task")
    stack = self._leases.get(task)
    if not stack:
      raise RuntimeError("guard context release mismatch")
    lease = stack.pop()
    if self._tpm_bucket is not None and not lease.committed:
      self._tpm_bucket.cancel(lease.reservation_id, time.time())
    if not stack:
      self._leases.pop(task, None)
    self.sem.release()

  def record_usage(
    self,
    lease: GuardLease | None,
    *,
    usage_prompt_tokens: int,
    usage_completion_tokens: int,
  ) -> float:
    if self._tpm_bucket is None:
      return 0.0
    total = max(0, usage_prompt_tokens) + max(0, usage_completion_tokens)
    reservation_id = lease.reservation_id if lease is not None else None
    now = time.time()
    wait = self._tpm_bucket.commit(reservation_id, total, now)
    if lease is not None:
      lease.committed = True
    return wait


class _GuardContext:
  def __init__(self, guard: "Guard", estimated_prompt_tokens: int):
    self._guard = guard
    self._estimated = estimated_prompt_tokens

  async def __aenter__(self) -> GuardLease:
    return await self._guard._acquire(self._estimated)

  async def __aexit__(self, exc_type, exc, tb):
    self._guard._release_current_task()


class ProviderGuards:
  def __init__(self, providers: Dict[str, ProviderDef]):
    self.guards: Dict[str, Guard] = {
      name: Guard(p.rpm, p.concurrency, p.tpm) for name, p in providers.items()
    }

  def get(self, name: str) -> Guard:
    return self.guards[name]
