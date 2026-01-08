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
    """Wrap a coroutine with an optional timeout.
    
    Args:
        coro: Coroutine to execute.
        timeout_seconds: Timeout in seconds.
        
    Returns:
        Result of the coroutine.
    """
    if timeout_seconds and timeout_seconds > 0:
        return await asyncio.wait_for(coro, timeout_seconds)
    return await coro


def is_retriable_error(exception: Exception) -> bool:
    """Check if an exception is retriable.
    
    Retriable errors include:
    - Timeout and cancellation errors
    - HTTP 5xx errors (server errors)
    - HTTP 429 (rate limit)
    - Connection errors
    
    Args:
        exception: Exception to check.
        
    Returns:
        True if the error should be retried.
    """
    # Always retry timeouts and cancellations
    if isinstance(exception, (asyncio.TimeoutError, asyncio.CancelledError)):
        return True
    
    # Check for Mistral SDK errors
    try:
        from mistralai.models.sdkerror import SDKError
        if isinstance(exception, SDKError):
            # Retry on 5xx errors and 429 (rate limit)
            if hasattr(exception, 'status_code'):
                status = exception.status_code
                return status >= 500 or status == 429
            # Also retry on connection errors (no status code)
            error_msg = str(exception).lower()
            return any(keyword in error_msg for keyword in [
                'connect error', 'disconnect', 'reset', 'overflow',
                'connection', 'timeout', 'unavailable'
            ])
    except ImportError:
        pass
    
    # Check for Google API errors
    try:
        from google.api_core.exceptions import GoogleAPIError
        if isinstance(exception, GoogleAPIError):
            # Retry on 5xx errors and 429
            if hasattr(exception, 'code'):
                return exception.code >= 500 or exception.code == 429
    except ImportError:
        pass
    
    # Check for generic HTTP errors
    error_msg = str(exception).lower()
    if any(keyword in error_msg for keyword in ['503', '502', '500', '429']):
        return True
    
    return False


async def request_with_retry(
    coro_factory,
    rate_limiter: AsyncRateLimiter,
    timeout_seconds: float,
    max_retries: int = 2,
    base_delay: float = 1.0
):
    """Execute a coroutine with retry logic and rate limiting.
    
    Retries on transient errors including timeouts, cancellations,
    and API errors (5xx, 429, connection errors). Each retry respects
    the rate limiter to maintain overall RPS limit.
    
    Args:
        coro_factory: Callable that returns a fresh coroutine for each attempt.
        rate_limiter: Rate limiter instance.
        timeout_seconds: Timeout for each attempt.
        max_retries: Maximum number of retry attempts (default: 2).
        base_delay: Base delay in seconds for exponential backoff (default: 1.0).
        
    Returns:
        Result of the coroutine.
        
    Raises:
        Last exception if all retries are exhausted.
    """
    import logging
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            # Wait for rate limiter before each attempt
            await rate_limiter.wait()
            
            # Execute request with timeout
            coro = coro_factory()
            return await request_with_timeout(coro, timeout_seconds)
            
        except Exception as e:
            # Check if error is retriable
            if not is_retriable_error(e):
                # Non-retriable error, raise immediately
                raise
            
            last_exception = e
            
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, etc.
                delay = base_delay * (2 ** attempt)
                logging.warning(
                    f"Retriable error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                # All retries exhausted
                logging.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                raise
    
    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
