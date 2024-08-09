import time
import asyncio

class RateLimiter:
    """Simple rate limiter to prevent overwhelming external services."""
    def __init__(self, rate: int, period: int):
        self.rate = rate
        self.period = period
        self.allowance = rate
        self.last_check = time.time()

    async def wait(self):
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.period)
        if self.allowance > self.rate:
            self.allowance = self.rate
        if self.allowance < 1:
            await asyncio.sleep((1 - self.allowance) * (self.period / self.rate))
            self.allowance = 0
        else:
            self.allowance -= 1