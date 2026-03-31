"""Optional Redis Streams publisher for real-time event fan-out."""

from typing import Sequence

import structlog
import redis

from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent

logger = structlog.get_logger()

STREAM_PITCHES = "mlb:pitches"
STREAM_AT_BATS = "mlb:at_bats"
STREAM_GAME_STATE = "mlb:game_state"


class RedisPublisher:
    """Publish pipeline events to Redis Streams. Best-effort — never crashes the pipeline."""

    def __init__(self, redis_client: redis.Redis, maxlen: int = 100_000):
        self._redis = redis_client
        self._maxlen = maxlen

    @classmethod
    def create(cls, redis_url: str, maxlen: int = 100_000) -> "RedisPublisher | None":
        """Try to connect to Redis. Returns None if unavailable."""
        try:
            client = redis.from_url(redis_url)
            client.ping()
            logger.info("redis_connected", url=redis_url)
            return cls(client, maxlen=maxlen)
        except (redis.ConnectionError, redis.TimeoutError, OSError) as exc:
            logger.warning("redis_unavailable", url=redis_url, error=str(exc))
            return None

    def publish_pitches(self, pitches: Sequence[PitchEvent]) -> None:
        """XADD each pitch to the pitches stream."""
        for p in pitches:
            try:
                self._redis.xadd(
                    STREAM_PITCHES,
                    {"data": p.model_dump_json()},
                    maxlen=self._maxlen,
                    approximate=True,
                )
            except Exception as exc:
                logger.warning("redis_publish_error", stream=STREAM_PITCHES, error=str(exc))

    def publish_at_bats(self, at_bats: Sequence[AtBatResult]) -> None:
        """XADD each at-bat to the at_bats stream."""
        for ab in at_bats:
            try:
                self._redis.xadd(
                    STREAM_AT_BATS,
                    {"data": ab.model_dump_json()},
                    maxlen=self._maxlen,
                    approximate=True,
                )
            except Exception as exc:
                logger.warning("redis_publish_error", stream=STREAM_AT_BATS, error=str(exc))

    def publish_game_state(self, event: GameStateEvent) -> None:
        """XADD a game state change to the game_state stream."""
        try:
            self._redis.xadd(
                STREAM_GAME_STATE,
                {"data": event.model_dump_json()},
                maxlen=self._maxlen,
                approximate=True,
            )
        except Exception as exc:
            logger.warning("redis_publish_error", stream=STREAM_GAME_STATE, error=str(exc))

    def close(self) -> None:
        try:
            self._redis.close()
        except Exception:
            pass
