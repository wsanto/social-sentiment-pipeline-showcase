"""
LunarCrush API Client

Fetches social media posts and metrics from LunarCrush API v4.
Focused on India-specific topics and content filtering.

Rate Limiting:
- Individual plan: 10 requests/minute
- Implements 6-second delay between requests to stay under limit
- Exponential backoff for 429 errors
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Rate limiting configuration
REQUESTS_PER_MINUTE = 10
REQUEST_DELAY_SECONDS = 60 / REQUESTS_PER_MINUTE  # 6 seconds between requests
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 10  # Initial backoff for 429 errors


class LunarCrushClient:
    """
    Async client for LunarCrush API v4

    Provides methods to fetch:
    - Topic lists
    - Posts by topic
    - Topic metadata and time series
    - News and creators
    """

    BASE_URL = "https://lunarcrush.com/api4"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LunarCrush client

        Args:
            api_key: LunarCrush API key (defaults to env var LUNARCRUSH_API_KEY)
        """
        self.api_key = api_key or os.getenv("LUNARCRUSH_API_KEY")
        if not self.api_key:
            raise ValueError("LunarCrush API key not provided")

        self.session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Rate limiting state
        self._last_request_time: Optional[float] = None
        self._request_lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self._headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure aiohttp session exists and is valid for current event loop"""
        try:
            # Check if session exists and is still valid
            if self.session is not None and not self.session.closed:
                # Verify the session's loop matches the current loop
                current_loop = asyncio.get_running_loop()
                if hasattr(self.session, '_loop') and self.session._loop is current_loop:
                    return  # Session is valid
                # Session is from a different loop, close it
                await self.session.close()
        except Exception:
            pass

        # Create a new session for the current event loop
        self.session = aiohttp.ClientSession(headers=self._headers)

    async def _wait_for_rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.
        Ensures minimum delay between requests to stay under API limits.
        """
        async with self._request_lock:
            if self._last_request_time is not None:
                elapsed = asyncio.get_event_loop().time() - self._last_request_time
                if elapsed < REQUEST_DELAY_SECONDS:
                    wait_time = REQUEST_DELAY_SECONDS - elapsed
                    logger.debug(
                        "rate_limit_wait",
                        wait_seconds=round(wait_time, 2)
                    )
                    await asyncio.sleep(wait_time)
            self._last_request_time = asyncio.get_event_loop().time()

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request to LunarCrush API with rate limiting and retry logic.

        Args:
            endpoint: API endpoint path (e.g., '/public/topics/list/v1')
            params: Query parameters
            retry_count: Current retry attempt (internal use)

        Returns:
            JSON response as dict

        Raises:
            aiohttp.ClientError: On HTTP errors after all retries exhausted
        """
        await self._ensure_session()

        # Enforce rate limiting before making request
        await self._wait_for_rate_limit()

        url = f"{self.BASE_URL}{endpoint}"

        logger.info(
            "lunarcrush_request",
            endpoint=endpoint,
            params=params,
            retry=retry_count
        )

        try:
            async with self.session.get(url, params=params) as response:
                # Handle rate limiting with backoff
                if response.status == 429:
                    if retry_count < MAX_RETRIES:
                        backoff_time = BASE_BACKOFF_SECONDS * (2 ** retry_count)
                        logger.warning(
                            "lunarcrush_rate_limited",
                            endpoint=endpoint,
                            retry=retry_count + 1,
                            backoff_seconds=backoff_time
                        )
                        await asyncio.sleep(backoff_time)
                        return await self._make_request(endpoint, params, retry_count + 1)
                    else:
                        logger.error(
                            "lunarcrush_rate_limit_exhausted",
                            endpoint=endpoint,
                            max_retries=MAX_RETRIES
                        )
                        response.raise_for_status()

                response.raise_for_status()
                data = await response.json()

                logger.info(
                    "lunarcrush_response",
                    endpoint=endpoint,
                    status=response.status
                )

                return data

        except aiohttp.ClientResponseError as e:
            logger.error(
                "lunarcrush_error",
                endpoint=endpoint,
                status=e.status,
                message=str(e)
            )
            raise
        except Exception as e:
            logger.error(
                "lunarcrush_unexpected_error",
                endpoint=endpoint,
                error=str(e)
            )
            raise

    async def get_topics_list(
        self,
        limit: int = 100,
        sort: str = "interactions_24h"
    ) -> List[Dict[str, Any]]:
        """
        Get list of trending topics

        Args:
            limit: Number of topics to return
            sort: Sort field (default: interactions_24h)

        Returns:
            List of topic objects with metadata
        """
        params = {
            "limit": limit,
            "sort": sort
        }

        response = await self._make_request("/public/topics/list/v1", params)
        return response.get("data", [])

    async def get_topic_details(self, topic: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific topic

        Args:
            topic: Topic name (lowercase, letters/numbers/spaces/#/$)

        Returns:
            Topic details with 24h aggregated metrics
        """
        endpoint = f"/public/topic/{topic}/v1"
        response = await self._make_request(endpoint)
        return response.get("data", {})

    async def get_topic_posts(
        self,
        topic: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 100,
        sort: str = "interactions"
    ) -> List[Dict[str, Any]]:
        """
        Get posts for a specific topic

        Args:
            topic: Topic name (lowercase)
            start: Start timestamp (Unix epoch)
            end: End timestamp (Unix epoch)
            limit: Max posts to return (default: 100)
            sort: Sort field (default: interactions)

        Returns:
            List of post objects
        """
        # Default to last 24 hours if no time range specified
        if start is None and end is None:
            end = int(datetime.utcnow().timestamp())
            start = int((datetime.utcnow() - timedelta(days=1)).timestamp())

        params = {
            "limit": limit,
            "sort": sort
        }

        if start:
            params["start"] = start
        if end:
            params["end"] = end

        endpoint = f"/public/topic/{topic}/posts/v1"
        response = await self._make_request(endpoint, params)
        return response.get("data", [])

    async def get_topic_news(
        self,
        topic: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get news articles for a topic

        Args:
            topic: Topic name
            start: Start timestamp
            end: End timestamp
            limit: Max news items to return

        Returns:
            List of news article objects
        """
        params = {"limit": limit}

        if start:
            params["start"] = start
        if end:
            params["end"] = end

        endpoint = f"/public/topic/{topic}/news/v1"
        response = await self._make_request(endpoint, params)
        return response.get("data", [])

    async def get_topic_creators(
        self,
        topic: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get influential creators for a topic

        Args:
            topic: Topic name
            limit: Max creators to return

        Returns:
            List of creator objects with engagement metrics
        """
        params = {"limit": limit}

        endpoint = f"/public/topic/{topic}/creators/v1"
        response = await self._make_request(endpoint, params)
        return response.get("data", [])

    async def get_topic_time_series(
        self,
        topic: str,
        bucket: str = "day",
        interval: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get time series data for a topic

        Args:
            topic: Topic name
            bucket: Time bucket ('hour' or 'day')
            interval: Convenience interval (e.g., '1w', '1m')
            start: Start timestamp
            end: End timestamp

        Returns:
            List of time series data points
        """
        params: Dict[str, Any] = {"bucket": bucket}

        if interval:
            params["interval"] = interval
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        endpoint = f"/public/topic/{topic}/time-series/v2"
        response = await self._make_request(endpoint, params)
        return response.get("data", [])

    async def fetch_india_posts(
        self,
        topics: List[str],
        hours_back: int = 24,
        limit_per_topic: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch posts for multiple India-specific topics.

        Processes topics SEQUENTIALLY with rate limiting to avoid 429 errors.
        With 18 topics and 6-second delays, this takes ~2 minutes to complete.

        Args:
            topics: List of topic names
            hours_back: How many hours back to fetch
            limit_per_topic: Max posts per topic

        Returns:
            Dict mapping topic -> list of posts
        """
        end = int(datetime.utcnow().timestamp())
        start = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp())

        results = {}
        successful = 0
        failed = 0

        logger.info(
            "fetch_india_posts_start",
            topic_count=len(topics),
            estimated_time_seconds=len(topics) * REQUEST_DELAY_SECONDS
        )

        # Process topics SEQUENTIALLY to respect rate limits
        # Rate limiting is handled in _make_request via _wait_for_rate_limit
        for i, topic in enumerate(topics):
            try:
                logger.info(
                    "fetching_topic",
                    topic=topic,
                    progress=f"{i+1}/{len(topics)}"
                )

                posts = await self.get_topic_posts(
                    topic=topic.lower(),
                    start=start,
                    end=end,
                    limit=limit_per_topic
                )
                results[topic] = posts
                successful += 1

                logger.info(
                    "fetched_topic_posts",
                    topic=topic,
                    post_count=len(posts)
                )
            except Exception as e:
                failed += 1
                logger.error(
                    "failed_fetch_topic",
                    topic=topic,
                    error=str(e)
                )
                results[topic] = []

        logger.info(
            "fetch_india_posts_complete",
            total_topics=len(topics),
            successful=successful,
            failed=failed
        )

        return results

    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()


# Convenience function for testing
async def test_client():
    """Test the LunarCrush client"""
    async with LunarCrushClient() as client:
        # Test topics list
        topics = await client.get_topics_list(limit=10)
        print(f"Found {len(topics)} topics")

        # Test fetching posts for a topic
        if topics:
            topic_name = topics[0].get("topic", "bitcoin")
            posts = await client.get_topic_posts(topic_name, limit=5)
            print(f"Found {len(posts)} posts for {topic_name}")


if __name__ == "__main__":
    asyncio.run(test_client())
