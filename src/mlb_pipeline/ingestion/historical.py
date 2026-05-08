"""Historical game loader — backfills completed games for a date range.

Unlike the live poller which diff-polls in-progress games, this fetches each
completed game's full live feed once and parses all events at once. Designed
for catching up after downtime or seeding the database from scratch.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date

import structlog

from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.ingestion.parser import parse_all_game_events
from mlb_pipeline.storage.sqlserver import SQLServerStorage

logger = structlog.get_logger()


@dataclass
class BackfillStats:
    games_found: int = 0
    games_processed: int = 0
    games_skipped: int = 0
    games_failed: int = 0
    games_not_final: int = 0
    pitches_stored: int = 0
    at_bats_stored: int = 0
    failed_game_pks: list[int] = field(default_factory=list)


class HistoricalGameLoader:
    """Backfills completed games over a date range into SQL Server."""

    def __init__(
        self,
        client: MLBStatsClient,
        storage: SQLServerStorage,
        concurrency: int = 8,
    ):
        self._client = client
        self._storage = storage
        self._sem = asyncio.Semaphore(concurrency)
        self._stats = BackfillStats()

    async def backfill(
        self,
        start: date,
        end: date,
        force: bool = False,
        sport_id: int = 1,
    ) -> BackfillStats:
        """Load all Final games in [start, end] inclusive.

        force=True re-ingests games that already have pitches in DB.
        """
        logger.info("backfill_starting", start=start.isoformat(), end=end.isoformat())

        games = await self._client.get_schedule_range(
            start.isoformat(), end.isoformat(), sport_id=sport_id
        )
        self._stats.games_found = len(games)
        logger.info("schedule_fetched", count=len(games))

        existing = self._storage.get_game_pks_with_pitches() if not force else set()

        tasks = []
        for game_data in games:
            game_pk = game_data["gamePk"]
            status = game_data["status"]["abstractGameState"]
            detailed = game_data["status"].get("detailedState", "")

            if status != "Final" or detailed in {"Postponed", "Cancelled", "Suspended"}:
                self._stats.games_not_final += 1
                logger.debug(
                    "skip_not_final",
                    game_pk=game_pk,
                    status=status,
                    detailed=detailed,
                )
                continue

            if game_pk in existing:
                self._stats.games_skipped += 1
                continue

            tasks.append(self._process_game(game_data))

        logger.info(
            "backfill_processing",
            to_process=len(tasks),
            skipped=self._stats.games_skipped,
            not_final=self._stats.games_not_final,
        )

        # Process in chunks so progress surfaces during long runs
        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i : i + chunk_size]
            await asyncio.gather(*chunk, return_exceptions=True)
            logger.info(
                "backfill_progress",
                processed=self._stats.games_processed,
                failed=self._stats.games_failed,
                total=len(tasks),
                pitches=self._stats.pitches_stored,
                at_bats=self._stats.at_bats_stored,
            )

        logger.info(
            "backfill_complete",
            games_found=self._stats.games_found,
            games_processed=self._stats.games_processed,
            games_skipped=self._stats.games_skipped,
            games_failed=self._stats.games_failed,
            pitches_stored=self._stats.pitches_stored,
            at_bats_stored=self._stats.at_bats_stored,
        )

        return self._stats

    async def _process_game(self, game_data: dict) -> None:
        async with self._sem:
            game_pk = game_data["gamePk"]
            try:
                feed = await self._client.get_live_feed(game_pk)
            except Exception as exc:
                self._stats.games_failed += 1
                self._stats.failed_game_pks.append(game_pk)
                logger.warning("feed_fetch_failed", game_pk=game_pk, error=str(exc))
                return

            try:
                pitches, at_bats = parse_all_game_events(game_pk, feed)
            except Exception as exc:
                self._stats.games_failed += 1
                self._stats.failed_game_pks.append(game_pk)
                logger.warning("parse_failed", game_pk=game_pk, error=str(exc))
                return

            try:
                self._upsert_game_from_feed(game_pk, game_data, feed)
                stored_pitches = await asyncio.to_thread(
                    self._storage.insert_pitches_batch, pitches
                )
                stored_at_bats = await asyncio.to_thread(
                    self._storage.insert_at_bats_batch, at_bats
                )
                self._stats.pitches_stored += stored_pitches
                self._stats.at_bats_stored += stored_at_bats
                self._stats.games_processed += 1
                logger.debug(
                    "game_loaded",
                    game_pk=game_pk,
                    pitches=stored_pitches,
                    at_bats=stored_at_bats,
                )
            except Exception as exc:
                self._stats.games_failed += 1
                self._stats.failed_game_pks.append(game_pk)
                logger.warning("storage_failed", game_pk=game_pk, error=str(exc))

    def _upsert_game_from_feed(
        self, game_pk: int, game_data: dict, feed: dict
    ) -> None:
        """Build a complete Game row from the live feed and upsert it."""
        from datetime import date as dt_date

        teams = game_data["teams"]
        away = teams["away"]
        home = teams["home"]
        game_date = dt_date.fromisoformat(game_data["gameDate"][:10])

        linescore = feed.get("liveData", {}).get("linescore", {})
        score_teams = linescore.get("teams", {})
        away_score = score_teams.get("away", {}).get("runs") or away.get("score") or 0
        home_score = score_teams.get("home", {}).get("runs") or home.get("score") or 0
        venue = (
            feed.get("gameData", {}).get("venue", {}).get("name")
            or game_data.get("venue", {}).get("name")
        )

        self._storage.upsert_game(
            game_pk=game_pk,
            game_date=game_date,
            game_type=game_data.get("gameType", "R"),
            season=int(game_data.get("season", game_date.year)),
            status="Final",
            away_team_id=away["team"]["id"],
            away_team_name=away["team"]["name"],
            home_team_id=home["team"]["id"],
            home_team_name=home["team"]["name"],
            venue_name=venue,
            away_score=away_score,
            home_score=home_score,
        )
