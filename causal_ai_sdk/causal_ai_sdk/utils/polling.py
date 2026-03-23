"""Polling logic utilities (async only)."""

import asyncio
from typing import Awaitable, Callable, Optional, Tuple, TypeVar

from causal_ai_sdk.exceptions import CausalAIError

T = TypeVar("T")


class PollingTimeoutError(CausalAIError):
    """Exception raised when polling times out."""

    pass


async def poll_until_ready_or_fail(
    check_func: Callable[[], Awaitable[T]],
    is_ready: Callable[[T], bool],
    is_failed: Callable[[T], bool],
    on_failed: Callable[[T], None],
    timeout: int = 300,
    interval: int = 3,
    timeout_error_message: Optional[str] = None,
    retry_exceptions: Optional[Tuple[type, ...]] = None,
    retry_if: Optional[Callable[[Exception], bool]] = None,
    on_poll: Optional[Callable[[float, T], None]] = None,
) -> T:
    """Poll until state is ready or failed; on failure call on_failed (should raise).

    Optionally retries check_func when it raises one of retry_exceptions and
    retry_if(exception) is True, until timeout.

    Args:
        check_func (Callable[[], Awaitable[T]]): Async function returning current state.
        is_ready (Callable[[T], bool]): True when state is successfully complete.
        is_failed (Callable[[T], bool]): True when state is a terminal failure.
        on_failed (Callable[[T], None]): Invoked when is_failed(state); should raise.
        timeout (int): Max wait in seconds.
        interval (int): Seconds between polls.
        timeout_error_message (Optional[str]): Message for PollingTimeoutError.
        retry_exceptions (Optional[Tuple[type, ...]]): Exception types to catch and retry.
        retry_if (Optional[Callable[[Exception], bool]]): Retry only when this returns True.
        on_poll (Optional[Callable[[float, T], None]]): If set, called each poll with
            (elapsed_sec, state).

    Returns:
        The final state when is_ready (type T).  # noqa: DAR003

    Raises:
        PollingTimeoutError: If timeout reached before ready/failed.
        Whatever on_failed(state) raises when is_failed.
    """
    loop = asyncio.get_running_loop()
    start_time = loop.time()
    message = timeout_error_message or f"Polling timed out after {timeout} seconds"

    while True:
        try:
            state = await check_func()
        except Exception as e:
            if retry_exceptions and isinstance(e, retry_exceptions) and retry_if and retry_if(e):
                elapsed = loop.time() - start_time
                if elapsed >= timeout:
                    raise PollingTimeoutError(message)
                await asyncio.sleep(interval)
                continue
            raise

        if is_failed(state):
            on_failed(state)
        if is_ready(state):
            return state

        elapsed = loop.time() - start_time
        if on_poll is not None:
            on_poll(elapsed, state)
        if elapsed >= timeout:
            raise PollingTimeoutError(message)
        await asyncio.sleep(interval)
