"""Advanced poller tests — storage failures, Redis integration, score tracking, run loop."""

import asyncio
import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlb_pipeline.config import Settings
from mlb_pipeline.ingestion.poller import LiveGamePoller, PollerStats, TrackedGame
from mlb_pipeline.models.enums import GameState
from mlb_pipeline.models.events import GameStateEvent

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def schedule_data():
    with open(FIXTURES_DIR / "schedule_sample.json") as f:
        return json.load(f)


@pytest.fixture
def live_feed_data():
    with open(FIXTURES_DIR / "live_feed_sample.json") as f:
        return json.load(f)


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.upsert_game = MagicMock()
    storage.insert_pitches_batch = MagicMock(return_value=0)
    storage.insert_at_bats_batch = MagicMock(return_value=0)
    storage.update_game_score = MagicMock()
    return storage


@pytest.fixture
def mock_client(schedule_data, live_feed_data):
    client = AsyncMock()
    client.get_schedule = AsyncMock(return_value=schedule_data)
    client.get_live_feed = AsyncMock(return_value=live_feed_data)
    return client


@pytest.fixture
def mock_publisher():
    pub = MagicMock()
    pub.publish_pitches = MagicMock()
    pub.publish_at_bats = MagicMock()
    pub.publish_game_state = MagicMock()
    return pub


@pytest.fixture
def test_settings():
    return Settings(
        db_server="localhost",
        db_name="test_db",
        poll_interval_live=0.01,
        poll_interval_idle=0.01,
        poll_interval_pregame=0.01,
    )


class TestStorageFailureRecovery:
    @pytest.mark.asyncio
    async def test_storage_failure_does_not_advance_tracking(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """If insert_pitches_batch fails, last_completed_play_index stays put."""
        mock_storage.insert_pitches_batch = MagicMock(side_effect=Exception("DB down"))

        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        # Tracking should NOT advance on failure
        assert game.last_completed_play_index == -1
        assert poller._stats.errors == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_storage_failure(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """After a failed poll, the next successful poll processes the same events."""
        call_count = 0

        def failing_then_ok(pitches):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("DB transient error")
            return len(pitches)

        mock_storage.insert_pitches_batch = MagicMock(side_effect=failing_then_ok)
        mock_storage.insert_at_bats_batch = MagicMock(return_value=0)

        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        # First poll fails
        await poller._poll_live_game(game)
        assert game.last_completed_play_index == -1

        # Second poll succeeds and reprocesses the same events
        await poller._poll_live_game(game)
        assert game.last_completed_play_index > -1

    @pytest.mark.asyncio
    async def test_game_upsert_failure_continues(
        self, mock_client, mock_storage, test_settings
    ):
        """If upsert_game throws for one game, other games still get registered."""
        call_count = 0
        original = mock_storage.upsert_game

        def fail_first(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("DB constraint violation")

        mock_storage.upsert_game = MagicMock(side_effect=fail_first)

        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        await poller._poll_schedule(date(2026, 3, 28))

        # Even though first upsert fails, games should still be tracked in memory
        assert len(poller._games) > 0


class TestRedisPublishingIntegration:
    @pytest.mark.asyncio
    async def test_pitches_published_to_redis(
        self, mock_client, mock_storage, mock_publisher, test_settings
    ):
        """New pitches should be published to Redis when publisher is present."""
        mock_storage.insert_pitches_batch = MagicMock(return_value=5)
        mock_storage.insert_at_bats_batch = MagicMock(return_value=1)

        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=mock_publisher,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        mock_publisher.publish_pitches.assert_called_once()
        mock_publisher.publish_at_bats.assert_called_once()

        pitches_arg = mock_publisher.publish_pitches.call_args[0][0]
        at_bats_arg = mock_publisher.publish_at_bats.call_args[0][0]
        assert len(pitches_arg) > 0
        assert len(at_bats_arg) > 0

    @pytest.mark.asyncio
    async def test_no_publish_when_no_new_events(
        self, mock_client, mock_storage, mock_publisher, test_settings, live_feed_data
    ):
        """When already caught up, nothing published."""
        all_plays = live_feed_data["liveData"]["plays"]["allPlays"]
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=mock_publisher,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            last_completed_play_index=len(all_plays) - 1,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        mock_publisher.publish_pitches.assert_not_called()
        mock_publisher.publish_at_bats.assert_not_called()

    @pytest.mark.asyncio
    async def test_state_change_publishes_game_state(
        self, mock_client, mock_storage, mock_publisher, test_settings, schedule_data
    ):
        """When a game's state changes during schedule poll, a GameStateEvent is published."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=mock_publisher,
            settings=test_settings,
        )

        # First poll — discover games
        await poller._poll_schedule(date(2026, 3, 28))
        first_game_pk = schedule_data["dates"][0]["games"][0]["gamePk"]
        tracked = poller._games[first_game_pk]
        original_state = tracked.state

        # Only test state change if original isn't Live
        if original_state != GameState.LIVE:
            modified = json.loads(json.dumps(schedule_data))
            modified["dates"][0]["games"][0]["status"]["abstractGameState"] = "Live"
            mock_client.get_schedule = AsyncMock(return_value=modified)

            await poller._poll_schedule(date(2026, 3, 28))

            mock_publisher.publish_game_state.assert_called_once()
            event_arg = mock_publisher.publish_game_state.call_args[0][0]
            assert isinstance(event_arg, GameStateEvent)
            assert event_arg.new_state == GameState.LIVE


class TestScoreTracking:
    @pytest.mark.asyncio
    async def test_score_updates_from_linescore(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """Poller should update TrackedGame scores from the linescore data."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        all_plays = live_feed_data["liveData"]["plays"]["allPlays"]
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            last_completed_play_index=len(all_plays) - 1,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        # The fixture is NYY 3 - SF 0
        linescore = live_feed_data["liveData"]["linescore"]["teams"]
        expected_away = linescore["away"]["runs"]
        expected_home = linescore["home"]["runs"]
        assert game.away_score == expected_away
        assert game.home_score == expected_home

    @pytest.mark.asyncio
    async def test_update_game_score_called(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """update_game_score should be called with the latest scores."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        all_plays = live_feed_data["liveData"]["plays"]["allPlays"]
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            last_completed_play_index=len(all_plays) - 1,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        mock_storage.update_game_score.assert_called_once()
        args = mock_storage.update_game_score.call_args[0]
        assert args[0] == 823243  # game_pk

    @pytest.mark.asyncio
    async def test_score_update_failure_does_not_crash(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """If update_game_score throws, the poller keeps going."""
        mock_storage.update_game_score = MagicMock(side_effect=Exception("DB error"))

        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        all_plays = live_feed_data["liveData"]["plays"]["allPlays"]
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            last_completed_play_index=len(all_plays) - 1,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        # Should not raise
        await poller._poll_live_game(game)


class TestGameStateFromLiveFeed:
    @pytest.mark.asyncio
    async def test_detects_final_from_live_feed(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """If the live feed says Final but game was Live, state should update."""
        # The fixture game is Final
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        feed_status = live_feed_data["gameData"]["status"]["abstractGameState"]
        if feed_status == "Final":
            assert game.state == GameState.FINAL

    @pytest.mark.asyncio
    async def test_state_change_from_feed_publishes_event(
        self, mock_client, mock_storage, mock_publisher, test_settings, live_feed_data
    ):
        """State transition detected from live feed should publish to Redis."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=mock_publisher,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        feed_status = live_feed_data["gameData"]["status"]["abstractGameState"]
        if feed_status != "Live":
            mock_publisher.publish_game_state.assert_called()


class TestPollerStats:
    def test_stats_initialized(self):
        stats = PollerStats()
        assert stats.games_discovered == 0
        assert stats.pitches_stored == 0
        assert stats.at_bats_stored == 0
        assert stats.api_calls == 0
        assert stats.errors == 0
        assert stats.started_at is not None

    @pytest.mark.asyncio
    async def test_api_calls_counted(
        self, mock_client, mock_storage, test_settings
    ):
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        await poller._poll_schedule(date(2026, 3, 28))
        assert poller._stats.api_calls == 1

        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
        )
        poller._games[823243] = game
        await poller._poll_live_game(game)
        assert poller._stats.api_calls == 2

    @pytest.mark.asyncio
    async def test_error_counter_resets_on_success(
        self, mock_client, mock_storage, test_settings
    ):
        """Per-game error counter resets when a poll succeeds."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243, state=GameState.LIVE,
            away_team="NYY", home_team="SF",
            errors=5,  # simulate prior errors
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        # Successful poll resets per-game error count
        assert game.errors == 0


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_run_exits_when_no_games(self, mock_storage, test_settings):
        """If no games found, run() returns immediately."""
        client = AsyncMock()
        client.get_schedule = AsyncMock(return_value={"dates": []})

        poller = LiveGamePoller(
            client=client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )

        # Should complete without hanging
        await asyncio.wait_for(poller.run(date(2026, 1, 15)), timeout=5.0)
        assert len(poller._games) == 0

    @pytest.mark.asyncio
    async def test_run_exits_when_all_final(
        self, mock_storage, test_settings, schedule_data, live_feed_data
    ):
        """Run loop exits when all games reach Final."""
        # Build a schedule with one game already Final
        single_game_schedule = {
            "dates": [{
                "games": [{
                    "gamePk": 823243,
                    "status": {"abstractGameState": "Final"},
                    "gameType": "R",
                    "season": "2026",
                    "teams": {
                        "away": {"team": {"id": 147, "name": "NYY"}},
                        "home": {"team": {"id": 137, "name": "SF"}},
                    },
                    "venue": {"name": "Oracle Park"},
                }]
            }]
        }

        client = AsyncMock()
        client.get_schedule = AsyncMock(return_value=single_game_schedule)

        poller = LiveGamePoller(
            client=client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )

        await asyncio.wait_for(poller.run(date(2026, 3, 28)), timeout=5.0)
        assert poller._games[823243].state == GameState.FINAL

    @pytest.mark.asyncio
    async def test_stop_halts_run_loop(self, mock_storage, test_settings):
        """Calling stop() causes the run loop to exit."""
        schedule = {
            "dates": [{
                "games": [{
                    "gamePk": 1,
                    "status": {"abstractGameState": "Preview"},
                    "gameType": "R",
                    "season": "2026",
                    "teams": {
                        "away": {"team": {"id": 147, "name": "NYY"}},
                        "home": {"team": {"id": 137, "name": "SF"}},
                    },
                    "venue": {"name": "Oracle Park"},
                }]
            }]
        }
        client = AsyncMock()
        client.get_schedule = AsyncMock(return_value=schedule)

        poller = LiveGamePoller(
            client=client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            await poller.stop()

        # Run both concurrently — stop should cause run to exit
        await asyncio.wait_for(
            asyncio.gather(poller.run(date(2026, 3, 28)), stop_after_delay()),
            timeout=5.0,
        )
