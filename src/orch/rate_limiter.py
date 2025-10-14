import asyncio
import time
from typing import Dict
from .router import ProviderDef

class TokenBucket:
  def __init__(self, rpm: int):
    self.capacity = max(1, rpm)
    self.tokens = self.capacity
    self.window_start = int(time.time() // 60)

  def try_take(self) -> float:
    # returns delay seconds if need wait, 0 if token taken
    now_min = int(time.time() // 60)
    if now_min != self.window_start:
      self.window_start = now_min
      self.tokens = self.capacity
    if self.tokens > 0:
      self.tokens -= 1
      return 0.0
    # time until next minute boundary
    now = time.time()
    return 60 - (now % 60)

class Guard:
  def __init__(self, rpm: int, concurrency: int):
    self.bucket = TokenBucket(rpm)
    self.sem = asyncio.Semaphore(concurrency)

  async def __aenter__(self):
    delay = self.bucket.try_take()
    while delay > 0:
      await asyncio.sleep(delay)
      delay = self.bucket.try_take()
    await self.sem.acquire()
    return self

  async def __aexit__(self, exc_type, exc, tb):
    self.sem.release()

class ProviderGuards:
  def __init__(self, providers: Dict[str, ProviderDef]):
    self.guards: Dict[str, Guard] = {name: Guard(p.rpm, p.concurrency) for name, p in providers.items()}

  def get(self, name: str) -> Guard:
    return self.guards[name]
