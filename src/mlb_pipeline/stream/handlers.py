"""Stream event handlers: storage, win-probability updates, and broadcasting.

Each handler is an async callable that receives a deserialized event and
performs a side effect (write to DB, run inference, broadcast via WebSocket).
"""

import structlog

from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent

logger = structlog.get_logger(__name__)


class StorageHandler:
    """Persists events to SQL Server via the async SQL writer."""

    def __init__(self, writer):
        self._writer = writer
        self.log = logger.bind(component="storage_handler")

    async def handle_pitch(self, event: PitchEvent) -> None:
        try:
            await self._writer.upsert_pitch(event)
        except Exception as exc:
            self.log.error("pitch_write_failed", event_id=event.event_id, error=str(exc))

    async def handle_at_bat(self, event: AtBatResult) -> None:
        try:
            await self._writer.upsert_at_bat(event)
        except Exception as exc:
            self.log.error(
                "at_bat_write_failed",
                game_pk=event.game_pk,
                at_bat=event.at_bat_index,
                error=str(exc),
            )

    async def handle_game_state(self, event: GameStateEvent) -> None:
        try:
            await self._writer.upsert_game(event)
        except Exception as exc:
            self.log.error("game_state_write_failed", game_pk=event.game_pk, error=str(exc))


class WinProbHandler:
    """Runs inference on each at-bat and publishes the result back to Redis."""

    def __init__(self, inference_service, producer):
        self._inference = inference_service
        self._producer = producer
        self.log = logger.bind(component="win_prob_handler")

    async def handle_at_bat(self, event: AtBatResult) -> None:
        try:
            prob = await self._inference.predict(event)
            await self._producer.publish_win_probability(
                game_pk=event.game_pk,
                home_win_prob=prob,
                metadata={
                    "inning": event.inning,
                    "half_inning": event.half_inning.value,
                    "away_score": event.away_score,
                    "home_score": event.home_score,
                    "outs": event.outs_after,
                },
            )
        except Exception as exc:
            self.log.warning("inference_failed", game_pk=event.game_pk, error=str(exc))


class BroadcastHandler:
    """Broadcasts events to all connected WebSocket clients."""

    def __init__(self, broadcast_fn):
        """
        Args:
            broadcast_fn: async callable(channel: str, payload: dict) -> None
        """
        self._broadcast = broadcast_fn
        self.log = logger.bind(component="broadcast_handler")

    async def handle_pitch(self, event: PitchEvent) -> None:
        try:
            await self._broadcast(
                f"game:{event.game_pk}:pitches",
                {
                    "type": "pitch",
                    "game_pk": event.game_pk,
                    "pitcher": event.pitcher_name,
                    "batter": event.batter_name,
                    "pitch_type": event.pitch_type,
                    "speed": event.start_speed,
                    "call": event.call_description,
                    "inning": event.inning,
                    "half": event.half_inning.value,
                    "balls": event.balls,
                    "strikes": event.strikes,
                    "outs": event.outs,
                    "away_score": event.away_score,
                    "home_score": event.home_score,
                },
            )
        except Exception as exc:
            self.log.warning("broadcast_pitch_failed", error=str(exc))

    async def handle_at_bat(self, event: AtBatResult) -> None:
        try:
            await self._broadcast(
                f"game:{event.game_pk}:at_bats",
                {
                    "type": "at_bat",
                    "game_pk": event.game_pk,
                    "event": event.event,
                    "description": event.description,
                    "batter": event.batter_name,
                    "pitcher": event.pitcher_name,
                    "rbi": event.rbi,
                    "away_score": event.away_score,
                    "home_score": event.home_score,
                    "is_scoring_play": event.is_scoring_play,
                },
            )
        except Exception as exc:
            self.log.warning("broadcast_at_bat_failed", error=str(exc))
