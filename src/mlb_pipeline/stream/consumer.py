"""Redis Streams consumer with consumer groups.

Reads events from mlb:pitches, mlb:at_bats, and mlb:game_state streams using
consumer groups for at-least-once delivery. Dispatches events to registered
handlers.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import redis.asyncio as aioredis
import structlog

from mlb_pipeline.config import settings
from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent
from mlb_pipeline.stream.producer import STREAM_AT_BATS, STREAM_GAME_STATE, STREAM_PITCHES

logger = structlog.get_logger(__name__)

PitchHandler = Callable[[PitchEvent], Coroutine[Any, Any, None]]
AtBatHandler = Callable[[AtBatResult], Coroutine[Any, Any, None]]
GameStateHandler = Callable[[GameStateEvent], Coroutine[Any, Any, None]]

GROUP_NAME = "mlb_pipeline"
CONSUMER_NAME = "worker_1"
BLOCK_MS = 2000  # block up to 2s waiting for new messages
BATCH_SIZE = 50


class EventConsumer:
    """Reads and dispatches events from Redis Streams consumer groups.

    Usage::

        async with EventConsumer() as consumer:
            consumer.on_pitch(storage_handler.handle_pitch)
            consumer.on_at_bat(storage_handler.handle_at_bat)
            await consumer.run()
    """

    def __init__(self, redis_url: str | None = None, consumer_name: str = CONSUMER_NAME):
        self._url = redis_url or settings.redis_url
        self._consumer_name = consumer_name
        self._redis: aioredis.Redis | None = None
        self._pitch_handlers: list[PitchHandler] = []
        self._at_bat_handlers: list[AtBatHandler] = []
        self._game_state_handlers: list[GameStateHandler] = []
        self._running = False
        self.log = logger.bind(component="consumer", name=consumer_name)

    async def __aenter__(self) -> "EventConsumer":
        self._redis = await aioredis.from_url(self._url, decode_responses=True)
        await self._ensure_groups()
        self.log.info("consumer_connected")
        return self

    async def __aexit__(self, *_) -> None:
        self._running = False
        if self._redis:
            await self._redis.aclose()

    def on_pitch(self, handler: PitchHandler) -> None:
        self._pitch_handlers.append(handler)

    def on_at_bat(self, handler: AtBatHandler) -> None:
        self._at_bat_handlers.append(handler)

    def on_game_state(self, handler: GameStateHandler) -> None:
        self._game_state_handlers.append(handler)

    async def run(self) -> None:
        """Blocking read loop. Call from an asyncio task."""
        self._running = True
        self.log.info("consumer_loop_started")
        while self._running:
            try:
                await self._read_batch()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("consumer_loop_error", error=str(exc))
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _ensure_groups(self) -> None:
        for stream in [STREAM_PITCHES, STREAM_AT_BATS, STREAM_GAME_STATE]:
            try:
                await self._redis.xgroup_create(stream, GROUP_NAME, id="$", mkstream=True)
                self.log.info("group_created", stream=stream)
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    async def _read_batch(self) -> None:
        streams = {
            STREAM_PITCHES: ">",
            STREAM_AT_BATS: ">",
            STREAM_GAME_STATE: ">",
        }
        results = await self._redis.xreadgroup(
            GROUP_NAME,
            self._consumer_name,
            streams,
            count=BATCH_SIZE,
            block=BLOCK_MS,
        )
        if not results:
            return

        for stream_name, messages in results:
            for msg_id, fields in messages:
                try:
                    await self._dispatch(stream_name, fields["data"])
                    await self._redis.xack(stream_name, GROUP_NAME, msg_id)
                except Exception as exc:
                    self.log.error(
                        "dispatch_failed",
                        stream=stream_name,
                        msg_id=msg_id,
                        error=str(exc),
                    )

    async def _dispatch(self, stream: str, data: str) -> None:
        if stream == STREAM_PITCHES:
            event = PitchEvent.model_validate_json(data)
            for h in self._pitch_handlers:
                await h(event)
        elif stream == STREAM_AT_BATS:
            event = AtBatResult.model_validate_json(data)
            for h in self._at_bat_handlers:
                await h(event)
        elif stream == STREAM_GAME_STATE:
            event = GameStateEvent.model_validate_json(data)
            for h in self._game_state_handlers:
                await h(event)
