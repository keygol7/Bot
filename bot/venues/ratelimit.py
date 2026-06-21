"""A small async token-bucket rate limiter for read polling.

Permits up to ``burst`` calls to proceed immediately, then paces the steady rate at
``rate_per_min``. Crucially — unlike a limiter that serializes every call behind a held
lock — the wait happens OUTSIDE the lock, so concurrent callers (e.g. an ``asyncio.gather``
prime sweep) actually overlap up to the burst. That is what makes a parallel sweep faster
than one-at-a-time; a serialize-on-a-lock limiter drains a gather strictly at the rate,
defeating the concurrency. One instance per venue so each respects its own read budget.
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    def __init__(self, rate_per_min: float, burst: int | None = None) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be > 0")
        self._rate = rate_per_min / 60.0                       # tokens per second
        # Default burst ~5s of tokens (>= the prime concurrency), so a parallel sweep fires
        # its batch at once and then settles to the steady rate.
        self._capacity = float(burst if burst is not None else max(1, round(rate_per_min / 12.0)))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                delay = (1.0 - self._tokens) / self._rate      # time until the next token
            await asyncio.sleep(delay)                          # OUTSIDE the lock -> concurrency
