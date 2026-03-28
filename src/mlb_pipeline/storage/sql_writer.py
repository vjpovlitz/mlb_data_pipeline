"""Async-friendly SQL Server writer using SQLAlchemy + threadpool execution.

SQLAlchemy's synchronous engine is wrapped in asyncio.to_thread so it doesn't
block the event loop. Each write is a small, index-friendly upsert.
"""

import asyncio
from contextlib import contextmanager

import structlog
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from mlb_pipeline.config import settings
from mlb_pipeline.models.database import AtBat, Game, Pitch, WinProbabilityLog
from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent

logger = structlog.get_logger(__name__)


class SQLWriter:
    """Thread-safe writer that wraps the sync SQLAlchemy engine.

    All public methods are async; heavy DB work runs in a thread executor.
    """

    def __init__(self, connection_string: str | None = None):
        self._conn_str = connection_string or settings.db_connection_string
        self._engine = create_engine(self._conn_str, pool_pre_ping=True, pool_size=5)
        self.log = logger.bind(component="sql_writer")

    @contextmanager
    def _session(self):
        with Session(self._engine) as session:
            yield session
            session.commit()

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def upsert_pitch(self, event: PitchEvent) -> None:
        await asyncio.to_thread(self._upsert_pitch_sync, event)

    async def upsert_at_bat(self, event: AtBatResult) -> None:
        await asyncio.to_thread(self._upsert_at_bat_sync, event)

    async def upsert_game(self, event: GameStateEvent) -> None:
        await asyncio.to_thread(self._upsert_game_sync, event)

    async def log_win_probability(
        self,
        game_pk: int,
        inning: int,
        half_inning: str,
        outs: int,
        away_score: int,
        home_score: int,
        runner_on_first: bool,
        runner_on_second: bool,
        runner_on_third: bool,
        probability: float,
        model_version: str,
    ) -> None:
        await asyncio.to_thread(
            self._log_win_prob_sync,
            game_pk, inning, half_inning, outs,
            away_score, home_score,
            runner_on_first, runner_on_second, runner_on_third,
            probability, model_version,
        )

    async def bulk_upsert_pitches(self, events: list[PitchEvent]) -> None:
        await asyncio.to_thread(self._bulk_pitches_sync, events)

    async def bulk_upsert_at_bats(self, events: list[AtBatResult]) -> None:
        await asyncio.to_thread(self._bulk_at_bats_sync, events)

    # ------------------------------------------------------------------
    # Sync helpers (run in thread executor)
    # ------------------------------------------------------------------

    def _upsert_pitch_sync(self, event: PitchEvent) -> None:
        with self._session() as session:
            row = Pitch(
                game_pk=event.game_pk,
                at_bat_index=event.at_bat_index,
                pitch_number=event.pitch_number,
                event_id=event.event_id,
                timestamp=event.timestamp.replace(tzinfo=None),
                inning=event.inning,
                half_inning=event.half_inning.value,
                pitcher_id=event.pitcher_id,
                pitcher_name=event.pitcher_name,
                batter_id=event.batter_id,
                batter_name=event.batter_name,
                pitch_type=event.pitch_type,
                start_speed=event.start_speed,
                end_speed=event.end_speed,
                spin_rate=event.spin_rate,
                spin_direction=event.spin_direction,
                break_angle=event.break_angle,
                break_length=event.break_length,
                pfx_x=event.pfx_x,
                pfx_z=event.pfx_z,
                plate_x=event.plate_x,
                plate_z=event.plate_z,
                zone=event.zone,
                call_code=event.call_code,
                call_description=event.call_description,
                is_in_play=event.is_in_play,
                is_strike=event.is_strike,
                is_ball=event.is_ball,
                launch_speed=event.launch_speed,
                launch_angle=event.launch_angle,
                total_distance=event.total_distance,
                trajectory=event.trajectory,
                balls=event.balls,
                strikes=event.strikes,
                outs=event.outs,
                runner_on_first=event.runner_on_first,
                runner_on_second=event.runner_on_second,
                runner_on_third=event.runner_on_third,
                away_score=event.away_score,
                home_score=event.home_score,
            )
            session.merge(row)

    def _upsert_at_bat_sync(self, event: AtBatResult) -> None:
        with self._session() as session:
            row = AtBat(
                game_pk=event.game_pk,
                at_bat_index=event.at_bat_index,
                timestamp=event.timestamp.replace(tzinfo=None),
                inning=event.inning,
                half_inning=event.half_inning.value,
                pitcher_id=event.pitcher_id,
                pitcher_name=event.pitcher_name,
                batter_id=event.batter_id,
                batter_name=event.batter_name,
                event=event.event,
                event_type=event.event_type,
                description=event.description,
                rbi=event.rbi,
                away_score=event.away_score,
                home_score=event.home_score,
                is_scoring_play=event.is_scoring_play,
                outs_after=event.outs_after,
                pitch_count=event.pitch_count,
                runner_on_first=event.runner_on_first,
                runner_on_second=event.runner_on_second,
                runner_on_third=event.runner_on_third,
            )
            session.merge(row)

    def _upsert_game_sync(self, event: GameStateEvent) -> None:
        from datetime import date

        with self._session() as session:
            row = Game(
                game_pk=event.game_pk,
                game_date=date.today(),
                game_type="R",
                season=date.today().year,
                status=event.new_state.value.capitalize(),
                away_team_id=0,
                away_team_name=event.away_team or "",
                home_team_id=0,
                home_team_name=event.home_team or "",
                away_score=event.away_score,
                home_score=event.home_score,
                home_team_won=(
                    event.home_score > event.away_score
                    if event.new_state.value == "final"
                    else None
                ),
            )
            session.merge(row)

    def _log_win_prob_sync(
        self,
        game_pk, inning, half_inning, outs,
        away_score, home_score,
        r1, r2, r3,
        probability, model_version,
    ) -> None:
        from datetime import UTC, datetime

        with self._session() as session:
            row = WinProbabilityLog(
                game_pk=game_pk,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                inning=inning,
                half_inning=half_inning,
                outs=outs,
                away_score=away_score,
                home_score=home_score,
                runner_on_first=r1,
                runner_on_second=r2,
                runner_on_third=r3,
                home_win_probability=probability,
                model_version=model_version,
            )
            session.add(row)

    def _bulk_pitches_sync(self, events: list[PitchEvent]) -> None:
        for event in events:
            self._upsert_pitch_sync(event)

    def _bulk_at_bats_sync(self, events: list[AtBatResult]) -> None:
        for event in events:
            self._upsert_at_bat_sync(event)

    def dispose(self) -> None:
        self._engine.dispose()
