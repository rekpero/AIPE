import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional

import aiohttp
from langchain.tools import BaseTool
from pydantic import Field

from config import Step
from utils.rate_limiter import RateLimiter

# Set up logging
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT = 5  # requests per second
RATE_LIMIT_PERIOD = 1  # second

rate_limiter = RateLimiter(RATE_LIMIT, RATE_LIMIT_PERIOD)

class WebhookTool(BaseTool):
    """Tool for calling webhooks with given parameters."""

    name: str = "CallWebhook"
    description: str = "Calls a webhook with given parameters."
    max_retries: int = Field(default=3)
    base_delay: float = Field(default=1.0)

    async def _call_webhook(self, url: str, method: str, payload: Dict[str, Any]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": os.environ.get("WEBHOOK_AUTH_TOKEN", "")  # Add if needed
        }

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    await rate_limiter.wait()
                    if method.upper() == "POST":
                        async with session.post(url, json=payload, headers=headers) as response:
                            response.raise_for_status()
                            return await response.text()
                    elif method.upper() == "GET":
                        async with session.get(url, params=payload, headers=headers) as response:
                            response.raise_for_status()
                            return await response.text()
                    else:
                        return f"Unsupported HTTP method: {method}"
                except aiohttp.ClientError as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == self.max_retries - 1:
                        return f"Error calling webhook after {self.max_retries} attempts: {str(e)}"
                    await asyncio.sleep(self.base_delay * (2 ** attempt))  # Exponential backoff

    async def _run(self, step: Step, context: Dict[str, Any]) -> str:
        url = step.url
        payload = step.payload
        method = step.method
        
        try:
            if method == "POST":
                response = await self._call_webhook(url, method, payload)
            elif method == "GET":
                response = await self._call_webhook(url, method, payload)
            else:
                return f"Unsupported HTTP method: {method}"
            
            return f"Webhook called successfully. Response: {response}"
        except Exception as e:
            return f"Error calling webhook: {str(e)}"
    
    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        # Async implementation, for now we can just call the sync version
        return await self._run(step, context)