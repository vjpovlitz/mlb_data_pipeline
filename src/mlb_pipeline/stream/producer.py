"""Redis Streams event producer.

Publishes PitchEvent, AtBatResult, and GameStateEvent messages to their
respective Redis Streams. Serializes via Pydantic JSON.
"""

import json

import redis.asyncio as aioredis
import structlog

from mlb_pipeline.config import settings
from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent

logger = structlog.get_logger(__name__)

STREAM_PITCHES = "mlb:pitches"
STREAM_AT_BATS = "mlb:at_bats"
STREAM_GAME_STATE = "mlb:game_state"

# Keep streams bounded — ~100k entries each
MAXLEN = 100_000


class EventProducer:
    """Async context manager that publishes events to Redis Streams."""

    def __init__(self, redis_url: str | None = None):
        self._url = redis_url or settings.redis_url
        self._redis: aioredis.Redis | None = None
        self.log = logger.bind(component="producer")

    async def __aenter__(self) -> "EventProducer":
        self._redis = await aioredis.from_url(self._url, decode_responses=True)
        self.log.info("producer_connected", url=self._url)
        return self

    async def __aexit__(self, *_) -> None:
        if self._redis:
            await self._redis.aclose()

    async def publish_pitch(self, event: PitchEvent) -> str:
        """Publish a PitchEvent to mlb:pitches stream. Returns stream entry ID."""
        entry_id = await self._redis.xadd(
            STREAM_PITCHES,
            {"data": event.model_dump_json()},
            maxlen=MAXLEN,
            approximate=True,
        )
        return entry_id

    async def publish_at_bat(self, event: AtBatResult) -> str:
        """Publish an AtBatResult to mlb:at_bats stream. Returns stream entry ID."""
        entry_id = await self._redis.xadd(
            STREAM_AT_BATS,
            {"data": event.model_dump_json()},
            maxlen=MAXLEN,
            approximate=True,
        )
        return entry_id

    async def publish_game_state(self, event: GameStateEvent) -> str:
        """Publish a GameStateEvent to mlb:game_state stream. Returns stream entry ID."""
        entry_id = await self._redis.xadd(
            STREAM_GAME_STATE,
            {"data": event.model_dump_json()},
            maxlen=MAXLEN,
            approximate=True,
        )
        self.log.info(
            "game_state_published",
            game_pk=event.game_pk,
            state=event.new_state.value,
        )
        return entry_id

    async def publish_win_probability(self, game_pk: int, home_win_prob: float, metadata: dict) -> str:
        """Publish a win probability update to mlb:win_prob stream."""
        payload = json.dumps({"game_pk": game_pk, "home_win_prob": home_win_prob, **metadata})
        entry_id = await self._redis.xadd(
            "mlb:win_prob",
            {"data": payload},
            maxlen=MAXLEN,
            approximate=True,
        )
        return entry_id

    async def get_stream_lengths(self) -> dict[str, int]:
        """Return current length of each stream for monitoring."""
        streams = [STREAM_PITCHES, STREAM_AT_BATS, STREAM_GAME_STATE, "mlb:win_prob"]
        lengths = {}
        for s in streams:
            try:
                lengths[s] = await self._redis.xlen(s)
            except Exception:
                lengths[s] = -1
        return lengths
