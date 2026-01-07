"""Rate limiting and timeout helpers for AI calls."""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Simple async rate limiter using a minimum interval."""

    def __init__(self, rps: float):
        self._rps = rps
        self._min_interval = 1.0 / rps if rps and rps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last_ts = 0.0

    async def wait(self) -> None:
        """Wait until the next request slot is available."""
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_ts
            remaining = self._min_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_ts = time.monotonic()


async def request_with_timeout(coro, timeout_seconds: float):
    """Wrap a coroutine with an optional timeout."""
    if timeout_seconds and timeout_seconds > 0:
        return await asyncio.wait_for(coro, timeout_seconds)
    return await coro
