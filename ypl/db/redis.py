import logging
import os

from upstash_redis.asyncio import Redis

upstash_redis_client: Redis | None = None


async def get_upstash_redis_client() -> Redis:
    global upstash_redis_client
    if upstash_redis_client is None:
        redis_url = os.getenv("UPSTASH_REDIS_URL")
        redis_token = os.getenv("UPSTASH_REDIS_TOKEN")
        if not (redis_url and redis_token):
            raise ValueError("UPSTASH_REDIS environment variables is not set")
        upstash_redis_client = Redis(url=redis_url, token=redis_token)
        try:
            print(await upstash_redis_client.ping())
        except Exception as e:
            logging.error(f"Error initializing Upstash Redis client: {str(e)}")
            raise
    return upstash_redis_client
