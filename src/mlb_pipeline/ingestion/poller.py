"""Live game poller with diff algorithm.

Polls the MLB Stats API for active games, computes diffs against the last seen
state, and publishes only new events to Redis Streams. Uses asyncio for
concurrent polling of multiple games.
"""

import asyncio
from datetime import date

import structlog

from mlb_pipeline.config import settings
from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.ingestion.parser import parse_all_game_events
from mlb_pipeline.models.enums import GameState
from mlb_pipeline.models.events import GameStateEvent
from mlb_pipeline.stream.producer import EventProducer

logger = structlog.get_logger(__name__)


class GamePoller:
    """Polls a single game and emits new events via the producer.

    Tracks the index of the last processed at-bat so only genuinely new
    events are published on each poll cycle.
    """

    def __init__(self, game_pk: int, producer: EventProducer, client: MLBStatsClient):
        self.game_pk = game_pk
        self.producer = producer
        self.client = client
        self._last_at_bat_index: int = -1
        self._game_state: str = "Preview"
        self.log = logger.bind(game_pk=game_pk)

    async def poll(self) -> bool:
        """Fetch the live feed, diff against last state, and publish new events.

        Returns True while the game is still active, False when it's Final.
        """
        try:
            feed = await self.client.get_live_feed(self.game_pk)
        except Exception as exc:
            self.log.warning("feed_fetch_failed", error=str(exc))
            return True  # keep polling — transient failure

        game_data = feed.get("gameData", {})
        status = game_data.get("status", {})
        new_state = status.get("abstractGameState", "Preview")

        # Detect state transitions
        if new_state != self._game_state:
            away = game_data.get("teams", {}).get("away", {}).get("name")
            home = game_data.get("teams", {}).get("home", {}).get("name")
            linescore = feed.get("liveData", {}).get("linescore", {})
            teams_ls = linescore.get("teams", {})
            event = GameStateEvent(
                game_pk=self.game_pk,
                timestamp=_now(),
                previous_state=GameState(self._game_state.lower()),
                new_state=GameState(new_state.lower()),
                away_team=away,
                home_team=home,
                away_score=teams_ls.get("away", {}).get("runs", 0),
                home_score=teams_ls.get("home", {}).get("runs", 0),
            )
            await self.producer.publish_game_state(event)
            self._game_state = new_state
            self.log.info("game_state_changed", new_state=new_state)

        if new_state == "Final":
            return False

        if new_state != "Live":
            return True  # pre-game; nothing to parse yet

        # Parse all events and filter to only new ones
        pitches, at_bats = parse_all_game_events(self.game_pk, feed)

        new_pitches = [p for p in pitches if p.at_bat_index > self._last_at_bat_index]
        new_at_bats = [a for a in at_bats if a.at_bat_index > self._last_at_bat_index]

        for pitch in new_pitches:
            await self.producer.publish_pitch(pitch)

        for at_bat in new_at_bats:
            await self.producer.publish_at_bat(at_bat)

        if new_at_bats:
            self._last_at_bat_index = max(a.at_bat_index for a in new_at_bats)
            self.log.debug(
                "published_events",
                pitches=len(new_pitches),
                at_bats=len(new_at_bats),
                last_ab=self._last_at_bat_index,
            )

        return True


class LiveGamePoller:
    """Orchestrates polling of all live MLB games for a given date.

    Discovers which games are in progress, starts per-game pollers, and
    removes pollers for finished games. Falls back to checking for new
    games if the schedule changes (doubleheaders, postponements).
    """

    def __init__(self, target_date: date | None = None):
        self.target_date = target_date or date.today()
        self._pollers: dict[int, GamePoller] = {}
        self._producer: EventProducer | None = None
        self._client: MLBStatsClient | None = None
        self.log = logger.bind(date=str(self.target_date))

    async def run(self) -> None:
        """Main polling loop. Runs until all games on target_date are Final."""
        async with MLBStatsClient() as client, EventProducer() as producer:
            self._client = client
            self._producer = producer
            self.log.info("poller_started")

            while True:
                await self._sync_game_roster()

                if not self._pollers:
                    self.log.info("no_active_games_sleeping")
                    await asyncio.sleep(settings.poll_interval_idle)
                    continue

                # Poll all active games concurrently
                tasks = [
                    asyncio.create_task(poller.poll(), name=f"poll_{gp}")
                    for gp, poller in list(self._pollers.items())
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                finished = []
                for (game_pk, _), result in zip(self._pollers.items(), results):
                    if isinstance(result, Exception):
                        self.log.error("poller_exception", game_pk=game_pk, error=str(result))
                    elif result is False:
                        finished.append(game_pk)

                for gp in finished:
                    self.log.info("game_finished", game_pk=gp)
                    del self._pollers[gp]

                if not self._pollers:
                    self.log.info("all_games_finished")
                    break

                await asyncio.sleep(settings.poll_interval_live)

    async def _sync_game_roster(self) -> None:
        """Add pollers for any newly discovered live/pregame games."""
        try:
            schedule = await self._client.get_schedule(str(self.target_date))
        except Exception as exc:
            self.log.warning("schedule_fetch_failed", error=str(exc))
            return

        dates = schedule.get("dates", [])
        if not dates:
            return

        for game in dates[0].get("games", []):
            game_pk = game["gamePk"]
            abstract_state = game["status"].get("abstractGameState", "Preview")
            if abstract_state == "Final":
                continue
            if game_pk not in self._pollers:
                self._pollers[game_pk] = GamePoller(game_pk, self._producer, self._client)
                self.log.info("tracking_game", game_pk=game_pk, state=abstract_state)


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)
