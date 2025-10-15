from __future__ import annotations
import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")

async def run_with_timeout(aw: Awaitable[T], timeout_seconds: float) -> T:
    """Await an awaitable with a timeout.
    Raises asyncio.TimeoutError on timeout.
    """
    return await asyncio.wait_for(aw, timeout_seconds)

