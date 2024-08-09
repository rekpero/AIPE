import os
import logging
import time
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus

import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from pydantic import Field

from config import Step
from utils.rate_limiter import RateLimiter

# Set up logging
logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT = 1  # requests per second
RATE_LIMIT_PERIOD = 1  # second

rate_limiter = RateLimiter(RATE_LIMIT, RATE_LIMIT_PERIOD)

class WebSearchTool(BaseTool):
    """Tool for performing web searches."""

    name = "WebSearch"
    description = "Useful for searching the web for information."

    async def _run(self, step: Step, context: Dict[str, Any]) -> str:
        logger.info(f"Executing WebSearch with query: {step.query}")
        query = step.query
        if step.search_type == 'serper':
            return await self.serper_search(query, step.num_results)
        else:
            return await self.duckduckgo_search(query)
    
    async def _arun(self, step: Step, context: Dict[str, Any]) -> str:
        return await self._run(step, context)

    async def duckduckgo_search(self, query: str, max_retries: int = 3, retry_delay: int = 5) -> str:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    await rate_limiter.wait()
                    async with session.get(url, headers=headers) as response:
                        if response.status == 202:
                            logger.info(f"Received 202 status. Retrying in {retry_delay} seconds...")
                            await asyncio.sleep(retry_delay)
                            continue

                        response.raise_for_status()
                        content = await response.text()

                    soup = BeautifulSoup(content, 'html.parser')
                    results = soup.find_all('div', class_='result__body')
                    
                    if not results:
                        logger.warning("No search results found. HTML content might have changed.")
                        return "No search results found."

                    search_results = []
                    for result in results[:5]:  # Limit to top 5 results
                        title = result.find('a', class_='result__a').text
                        snippet = result.find('a', class_='result__snippet').text
                        link = result.find('a', class_='result__a')['href']
                        search_results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
                    
                    return "\n".join(search_results)
                except aiohttp.ClientError as e:
                    logger.error(f"An error occurred during the DuckDuckGo search (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        return f"An error occurred during the DuckDuckGo search after {max_retries} attempts: {str(e)}"
                    await asyncio.sleep(retry_delay)

    async def serper_search(self, query: str, num_results: int) -> str:
        url = "https://google.serper.dev/search"
        payload = {"q": query, "num": num_results}
        headers = {
            'X-API-KEY': os.environ.get('SERPER_API_KEY', ''),
            'Content-Type': 'application/json'
        }
        
        if not headers['X-API-KEY']:
            logger.error("SERPER_API_KEY environment variable is not set")
            return "Error: SERPER_API_KEY is not set"

        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                try:
                    await rate_limiter.wait()
                    async with session.post(url, json=payload, headers=headers) as response:
                        response.raise_for_status()
                        data = await response.json()

                    organic_results = data.get('organic', [])
                    
                    search_results = []
                    for result in organic_results:
                        title = result.get('title', 'No title')
                        snippet = result.get('snippet', 'No snippet')
                        link = result.get('link', 'No link')
                        search_results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")
                    
                    return "\n".join(search_results)
                except aiohttp.ClientError as e:
                    logger.error(f"An error occurred during the Serper search: {str(e)}")
                    if attempt == 2:
                        return f"Failed to perform Serper search after 3 attempts: {str(e)}"
                    await asyncio.sleep(1)

class WebScrapeTool(BaseTool):
    """Tool for scraping content from given URLs."""

    name = "WebScrape"
    description = "Useful for scraping content from given URLs."

    async def scrape_url(self, session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
        try:
            async with session.get(url, timeout=10) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                return url, text[:10000]  # Limit to first 10000 characters
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {str(e)}")
            return url, f"Failed to scrape {url}: {str(e)}"

    async def scrape_urls(self, urls: List[str]) -> Dict[str, str]:
        async with aiohttp.ClientSession() as session:
            tasks = [self.scrape_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
        return dict(results)

    async def _run(self, step: Step, context: Dict[str, Any]) -> Dict[str, str]:
        urls = step.urls
        return await self.scrape_urls(urls)
    
    async def _arun(self, step: Step, context: Dict[str, Any]) -> Dict[str, str]:
        return await self._run(step, context)