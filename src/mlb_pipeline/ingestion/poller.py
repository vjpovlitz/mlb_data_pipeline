"""Live game poller — the heart of the real-time pipeline.

Polls MLB Stats API for live game data, detects new pitches via diff algorithm,
and stores events to SQL Server (with optional Redis streaming).
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import structlog

from mlb_pipeline.config import Settings
from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.ingestion.parser import (
    parse_at_bat_result,
    parse_pitch_events,
)
from mlb_pipeline.models.enums import GameState
from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent
from mlb_pipeline.storage.sqlserver import SQLServerStorage
from mlb_pipeline.stream.publisher import RedisPublisher

logger = structlog.get_logger()


@dataclass
class TrackedGame:
    """In-memory state for a game being tracked by the poller."""

    game_pk: int
    state: GameState
    last_completed_play_index: int = -1
    current_ab_pitch_count: int = 0
    away_team: str = ""
    home_team: str = ""
    away_score: int = 0
    home_score: int = 0
    polls: int = 0
    errors: int = 0


@dataclass
class PollerStats:
    """Cumulative stats for the poller session."""

    games_discovered: int = 0
    pitches_stored: int = 0
    at_bats_stored: int = 0
    api_calls: int = 0
    errors: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LiveGamePoller:
    """Polls MLB Stats API for live game data and stores events."""

    def __init__(
        self,
        client: MLBStatsClient,
        storage: SQLServerStorage,
        publisher: RedisPublisher | None,
        settings: Settings,
    ):
        self._client = client
        self._storage = storage
        self._publisher = publisher
        self._settings = settings
        self._games: dict[int, TrackedGame] = {}
        self._stats = PollerStats()
        self._running = False

    async def run(self, target_date: date | None = None) -> None:
        """Main loop. Runs until cancelled or all games are Final."""
        target = target_date or date.today()
        self._running = True

        logger.info("poller_starting", date=target.isoformat())

        try:
            await self._poll_schedule(target)

            if not self._games:
                logger.info("no_games_found", date=target.isoformat())
                return

            schedule_poll_counter = 0

            while self._running:
                live_games = [
                    g for g in self._games.values() if g.state == GameState.LIVE
                ]

                if live_games:
                    tasks = [self._poll_live_game(g) for g in live_games]
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Check if all games are done
                all_done = all(
                    g.state in (GameState.FINAL, GameState.POSTPONED, GameState.SUSPENDED)
                    for g in self._games.values()
                )
                if all_done:
                    logger.info(
                        "all_games_final",
                        games=len(self._games),
                        pitches=self._stats.pitches_stored,
                        at_bats=self._stats.at_bats_stored,
                    )
                    break

                interval = self._determine_poll_interval()
                await asyncio.sleep(interval)

                # Re-poll schedule periodically to catch state transitions
                schedule_poll_counter += 1
                if not live_games or schedule_poll_counter >= 30:
                    await self._poll_schedule(target)
                    schedule_poll_counter = 0

        finally:
            self._log_status()

    async def _poll_schedule(self, target_date: date) -> None:
        """Discover games for the date and register them."""
        try:
            schedule = await self._client.get_schedule(target_date.isoformat())
            self._stats.api_calls += 1
        except Exception as exc:
            logger.error("schedule_poll_failed", error=str(exc))
            self._stats.errors += 1
            return

        for date_entry in schedule.get("dates", []):
            for game_data in date_entry.get("games", []):
                game_pk = game_data["gamePk"]
                abstract_state = game_data["status"]["abstractGameState"]

                try:
                    new_state = GameState(abstract_state)
                except ValueError:
                    logger.warning("unknown_game_state", state=abstract_state, game_pk=game_pk)
                    continue

                teams = game_data["teams"]
                away_name = teams["away"]["team"]["name"]
                home_name = teams["home"]["team"]["name"]
                away_id = teams["away"]["team"]["id"]
                home_id = teams["home"]["team"]["id"]

                # Upsert game to DB
                try:
                    await asyncio.to_thread(
                        self._storage.upsert_game,
                        game_pk=game_pk,
                        game_date=target_date,
                        game_type=game_data.get("gameType", "R"),
                        season=int(game_data.get("season", target_date.year)),
                        status=abstract_state,
                        away_team_id=away_id,
                        away_team_name=away_name,
                        home_team_id=home_id,
                        home_team_name=home_name,
                        venue_name=game_data.get("venue", {}).get("name"),
                    )
                except Exception as exc:
                    logger.error("game_upsert_failed", game_pk=game_pk, error=str(exc))

                if game_pk not in self._games:
                    self._games[game_pk] = TrackedGame(
                        game_pk=game_pk,
                        state=new_state,
                        away_team=away_name,
                        home_team=home_name,
                    )
                    self._stats.games_discovered += 1
                    logger.info(
                        "game_discovered",
                        game_pk=game_pk,
                        matchup=f"{away_name} @ {home_name}",
                        state=new_state.value,
                    )
                else:
                    tracked = self._games[game_pk]
                    old_state = tracked.state
                    if old_state != new_state:
                        logger.info(
                            "game_state_changed",
                            game_pk=game_pk,
                            matchup=f"{tracked.away_team} @ {tracked.home_team}",
                            old=old_state.value,
                            new=new_state.value,
                        )
                        if self._publisher:
                            event = GameStateEvent(
                                game_pk=game_pk,
                                timestamp=datetime.now(timezone.utc),
                                previous_state=old_state,
                                new_state=new_state,
                                away_team=tracked.away_team,
                                home_team=tracked.home_team,
                                away_score=tracked.away_score,
                                home_score=tracked.home_score,
                            )
                            self._publisher.publish_game_state(event)
                        tracked.state = new_state

    async def _poll_live_game(self, game: TrackedGame) -> None:
        """Fetch live feed for one game, detect new plays, store them."""
        try:
            feed = await self._client.get_live_feed(game.game_pk)
            self._stats.api_calls += 1
            game.polls += 1
        except Exception as exc:
            game.errors += 1
            self._stats.errors += 1
            logger.error(
                "live_feed_failed",
                game_pk=game.game_pk,
                error=str(exc),
                consecutive_errors=game.errors,
            )
            return

        # Reset error counter on success
        game.errors = 0

        all_plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])

        new_pitches: list[PitchEvent] = []
        new_at_bats: list[AtBatResult] = []

        # Process newly completed plays
        i = game.last_completed_play_index + 1
        while i < len(all_plays):
            play = all_plays[i]
            if not play.get("about", {}).get("isComplete", False):
                break
            pitches = parse_pitch_events(game.game_pk, play)
            at_bat = parse_at_bat_result(game.game_pk, play)
            new_pitches.extend(pitches)
            new_at_bats.append(at_bat)
            i += 1

        completed_up_to = i - 1

        # Process new pitches from in-progress at-bat
        in_progress_index = game.last_completed_play_index + 1
        if completed_up_to > game.last_completed_play_index:
            # Completed plays were found — in-progress is now the next one
            in_progress_index = completed_up_to + 1
            game.current_ab_pitch_count = 0

        if in_progress_index < len(all_plays):
            current_play = all_plays[in_progress_index]
            actual_pitches = [
                e for e in current_play.get("playEvents", []) if e.get("isPitch", False)
            ]
            new_pitch_count = len(actual_pitches) - game.current_ab_pitch_count
            if new_pitch_count > 0:
                all_pitches_in_ab = parse_pitch_events(game.game_pk, current_play)
                incremental = all_pitches_in_ab[game.current_ab_pitch_count :]
                new_pitches.extend(incremental)
                game.current_ab_pitch_count = len(actual_pitches)

        # Store to SQL Server
        if new_pitches or new_at_bats:
            try:
                stored_pitches = await asyncio.to_thread(
                    self._storage.insert_pitches_batch, new_pitches
                )
                stored_at_bats = await asyncio.to_thread(
                    self._storage.insert_at_bats_batch, new_at_bats
                )
                self._stats.pitches_stored += stored_pitches
                self._stats.at_bats_stored += stored_at_bats

                # Advance tracking state only after successful storage
                if completed_up_to > game.last_completed_play_index:
                    game.last_completed_play_index = completed_up_to

                if new_pitches:
                    last = new_pitches[-1]
                    logger.info(
                        "new_events",
                        game_pk=game.game_pk,
                        matchup=f"{game.away_team} @ {game.home_team}",
                        pitches=len(new_pitches),
                        at_bats=len(new_at_bats),
                        inning=f"{'T' if last.half_inning.value == 'top' else 'B'}{last.inning}",
                        pitcher=last.pitcher_name,
                        batter=last.batter_name,
                    )
            except Exception as exc:
                logger.error(
                    "storage_failed",
                    game_pk=game.game_pk,
                    error=str(exc),
                    pitches=len(new_pitches),
                    at_bats=len(new_at_bats),
                )
                self._stats.errors += 1
                return  # Don't advance tracking — retry on next poll

            # Publish to Redis (best effort)
            if self._publisher:
                if new_pitches:
                    self._publisher.publish_pitches(new_pitches)
                if new_at_bats:
                    self._publisher.publish_at_bats(new_at_bats)

        # Update game score from linescore
        linescore = feed.get("liveData", {}).get("linescore", {})
        teams_score = linescore.get("teams", {})
        away_score = teams_score.get("away", {}).get("runs", 0) or 0
        home_score = teams_score.get("home", {}).get("runs", 0) or 0
        game.away_score = away_score
        game.home_score = home_score

        game_status = (
            feed.get("gameData", {}).get("status", {}).get("abstractGameState", "Live")
        )
        if game_status != game.state.value:
            old_state = game.state
            try:
                game.state = GameState(game_status)
            except ValueError:
                pass
            else:
                logger.info(
                    "game_state_changed",
                    game_pk=game.game_pk,
                    old=old_state.value,
                    new=game.state.value,
                    score=f"{away_score}-{home_score}",
                )
                if self._publisher:
                    event = GameStateEvent(
                        game_pk=game.game_pk,
                        timestamp=datetime.now(timezone.utc),
                        previous_state=old_state,
                        new_state=game.state,
                        away_team=game.away_team,
                        home_team=game.home_team,
                        away_score=away_score,
                        home_score=home_score,
                    )
                    self._publisher.publish_game_state(event)

        try:
            await asyncio.to_thread(
                self._storage.update_game_score,
                game.game_pk,
                away_score,
                home_score,
                game_status,
            )
        except Exception as exc:
            logger.warning("score_update_failed", game_pk=game.game_pk, error=str(exc))

    def _determine_poll_interval(self) -> float:
        """Return the appropriate poll interval based on game states."""
        states = {g.state for g in self._games.values()}
        if GameState.LIVE in states:
            return self._settings.poll_interval_live
        elif GameState.PREVIEW in states:
            return self._settings.poll_interval_pregame
        else:
            return self._settings.poll_interval_idle

    def _log_status(self) -> None:
        """Log a status summary."""
        elapsed = (datetime.now(timezone.utc) - self._stats.started_at).total_seconds()
        logger.info(
            "poller_status",
            elapsed_seconds=round(elapsed),
            games=len(self._games),
            pitches_stored=self._stats.pitches_stored,
            at_bats_stored=self._stats.at_bats_stored,
            api_calls=self._stats.api_calls,
            errors=self._stats.errors,
        )
        for g in self._games.values():
            logger.info(
                "game_status",
                game_pk=g.game_pk,
                matchup=f"{g.away_team} @ {g.home_team}",
                state=g.state.value,
                score=f"{g.away_score}-{g.home_score}",
                polls=g.polls,
            )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
