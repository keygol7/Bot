"""A minimal async rate limiter for read polling.

Paces calls to at most ``rate_per_min`` by sleeping out the remaining interval since
the previous call. Standard-library only; one instance per venue so each respects
its own published read budget (e.g. Kalshi ~60 req/min on public reads).
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    def __init__(self, rate_per_min: float) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be > 0")
        self.min_interval = 60.0 / rate_per_min
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self._last = time.monotonic()
