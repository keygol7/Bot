"""Token-bucket rate limiter: bursts up to capacity, then paces; concurrency-safe."""

import asyncio
import time

import pytest

from bot.venues.ratelimit import AsyncRateLimiter


def test_burst_then_paces():
    # burst=3 at 600/min (10/s): first 3 immediate, the next 2 paced ~0.1s apart.
    lim = AsyncRateLimiter(rate_per_min=600, burst=3)

    async def run():
        t0 = time.monotonic()
        for _ in range(3):
            await lim.wait()
        burst = time.monotonic() - t0
        await lim.wait()
        await lim.wait()
        return burst, time.monotonic() - t0

    burst, total = asyncio.run(run())
    assert burst < 0.05            # the burst didn't block
    assert total >= 0.15          # 2 more paced at ~0.1s each (10/s)


def test_concurrent_callers_overlap():
    # gather of `burst` waits: all proceed at once because the sleep is OUTSIDE the lock.
    # A serialize-on-a-lock limiter would take ~0.4s here (paced at 10/s); the bucket ~0.
    lim = AsyncRateLimiter(rate_per_min=600, burst=5)

    async def run():
        t0 = time.monotonic()
        await asyncio.gather(*(lim.wait() for _ in range(5)))
        return time.monotonic() - t0

    assert asyncio.run(run()) < 0.1


def test_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        AsyncRateLimiter(0)
