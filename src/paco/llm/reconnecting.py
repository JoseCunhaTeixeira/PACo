"""A model server asked again while it refuses or drops the connection (a tunnel or a proxy in front
of it restarting): the waits doubling from 1 s, for a budget; a timeout not asked again, the server
was there. A function of the request to make, for both clients (paco.llm.model, paco.llm.claude)."""

import logging
from collections.abc import Awaitable, Callable

from openai import APIConnectionError, APITimeoutError

logger = logging.getLogger(__name__)


async def reconnecting[T](
    create: Callable[[], Awaitable[T]],
    server: object,
    reconnect_s: float,
    sleep: Callable[[float], Awaitable[None]],
) -> T:
    """`create`'s result, asked again while the model server at `server` refuses or drops the
    connection, the waits doubling from 1 s, for `reconnect_s` at most; then its error. A timeout
    is not asked again: the server was there."""
    waited, wait = 0.0, 1.0
    while True:
        try:
            return await create()
        except APITimeoutError:
            raise
        except APIConnectionError as error:
            left = reconnect_s - waited
            if left <= 0:
                raise
            pause = min(wait, left)
            logger.warning(
                "The model server at %s did not answer (%s): asked again in %g s",
                server,
                error,
                pause,
            )
            await sleep(pause)
            waited, wait = waited + pause, wait * 2
