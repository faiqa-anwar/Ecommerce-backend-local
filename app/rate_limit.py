"""
Minimal in-memory fixed-window rate limiter, keyed per store.

This is intentionally simple: it lives in one process's memory, so it
resets on restart and doesn't share state across multiple app instances.
That's fine for a single-instance deployment or local dev. The moment you
run more than one worker/replica, swap this for a Redis-backed limiter
(e.g. via slowapi + redis) so all instances share the same counters.
"""
import time
from collections import defaultdict, deque

from app.config import settings


class RateLimitExceeded(Exception):
    pass


class _FixedWindowLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        window_start = now - 60
        hits = self._hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= self.limit:
            raise RateLimitExceeded(f"Rate limit exceeded: max {self.limit} requests/minute.")
        hits.append(now)


chat_rate_limiter = _FixedWindowLimiter(settings.CHAT_RATE_LIMIT_PER_MINUTE)
