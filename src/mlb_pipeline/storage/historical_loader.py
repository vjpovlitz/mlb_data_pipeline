"""Historical data loader.

Fetches completed games from the MLB Stats API for a date range, parses them
into events, writes to both SQL Server and Parquet, and optionally replays
them through Redis Streams for pipeline testing.
"""

import asyncio
from datetime import date, timedelta

import structlog

from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.ingestion.parser import parse_all_game_events
from mlb_pipeline.storage.parquet_store import ParquetStore
from mlb_pipeline.storage.sql_writer import SQLWriter
from mlb_pipeline.stream.producer import EventProducer

logger = structlog.get_logger(__name__)


class HistoricalLoader:
    """Loads multiple games from the MLB API and persists them locally.

    Used for building training datasets before live pipeline is running.
    """

    def __init__(
        self,
        writer: SQLWriter | None = None,
        parquet: ParquetStore | None = None,
    ):
        self._writer = writer or SQLWriter()
        self._parquet = parquet or ParquetStore()
        self.log = logger.bind(component="historical_loader")

    async def load_date_range(
        self,
        start: date,
        end: date,
        *,
        write_sql: bool = True,
        write_parquet: bool = True,
        max_concurrent: int = 4,
    ) -> dict[str, int]:
        """Load all Final games between start and end (inclusive).

        Returns summary dict with counts.
        """
        async with MLBStatsClient() as client:
            game_pks = await self._collect_game_pks(client, start, end)
            self.log.info("games_to_load", count=len(game_pks), start=str(start), end=str(end))

            sem = asyncio.Semaphore(max_concurrent)
            tasks = [
                self._load_one_game(client, gp, write_sql=write_sql, write_parquet=write_parquet, sem=sem)
                for gp in game_pks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        success = sum(1 for r in results if r is True)
        failed = len(results) - success
        self.log.info("load_complete", success=success, failed=failed)
        return {"total": len(game_pks), "success": success, "failed": failed}

    async def load_single_game(
        self,
        game_pk: int,
        *,
        write_sql: bool = True,
        write_parquet: bool = True,
    ) -> tuple[int, int]:
        """Load a single completed game. Returns (pitch_count, at_bat_count)."""
        async with MLBStatsClient() as client:
            return await self._fetch_and_store(
                client, game_pk, write_sql=write_sql, write_parquet=write_parquet
            )

    async def replay_game(
        self,
        game_pk: int,
        *,
        speed: float = 1.0,
        producer: EventProducer | None = None,
    ) -> None:
        """Replay a game through Redis Streams for testing.

        If producer is None, creates its own connection.
        """
        async with MLBStatsClient() as client:
            feed = await client.get_live_feed(game_pk)

        pitches, at_bats = parse_all_game_events(game_pk, feed)

        async def _do_replay(prod: EventProducer) -> None:
            for at_bat in at_bats:
                ab_pitches = [p for p in pitches if p.at_bat_index == at_bat.at_bat_index]
                for pitch in ab_pitches:
                    await prod.publish_pitch(pitch)
                    if speed > 0:
                        await asyncio.sleep(0.1 / speed)
                await prod.publish_at_bat(at_bat)
                if speed > 0:
                    await asyncio.sleep(0.5 / speed)

        if producer is not None:
            await _do_replay(producer)
        else:
            async with EventProducer() as prod:
                await _do_replay(prod)

        self.log.info("replay_complete", game_pk=game_pk, pitches=len(pitches), at_bats=len(at_bats))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _collect_game_pks(self, client: MLBStatsClient, start: date, end: date) -> list[int]:
        game_pks: list[int] = []
        current = start
        while current <= end:
            try:
                schedule = await client.get_schedule(str(current))
                for date_entry in schedule.get("dates", []):
                    for game in date_entry.get("games", []):
                        if game["status"].get("abstractGameState") == "Final":
                            game_pks.append(game["gamePk"])
            except Exception as exc:
                self.log.warning("schedule_fetch_failed", date=str(current), error=str(exc))
            current += timedelta(days=1)
        return game_pks

    async def _load_one_game(
        self,
        client: MLBStatsClient,
        game_pk: int,
        *,
        write_sql: bool,
        write_parquet: bool,
        sem: asyncio.Semaphore,
    ) -> bool:
        async with sem:
            try:
                await self._fetch_and_store(client, game_pk, write_sql=write_sql, write_parquet=write_parquet)
                return True
            except Exception as exc:
                self.log.error("game_load_failed", game_pk=game_pk, error=str(exc))
                return False

    async def _fetch_and_store(
        self,
        client: MLBStatsClient,
        game_pk: int,
        *,
        write_sql: bool,
        write_parquet: bool,
    ) -> tuple[int, int]:
        feed = await client.get_live_feed(game_pk)
        game_data = feed.get("gameData", {})
        season = game_data.get("game", {}).get("season", str(date.today().year))
        season_int = int(season)

        pitches, at_bats = parse_all_game_events(game_pk, feed)

        if write_sql and pitches:
            await self._writer.bulk_upsert_pitches(pitches)
            await self._writer.bulk_upsert_at_bats(at_bats)

        if write_parquet and pitches:
            await asyncio.to_thread(self._parquet.write_pitches, game_pk, season_int, pitches)
            await asyncio.to_thread(self._parquet.write_at_bats, game_pk, season_int, at_bats)

        self.log.info("game_loaded", game_pk=game_pk, pitches=len(pitches), at_bats=len(at_bats))
        return len(pitches), len(at_bats)
