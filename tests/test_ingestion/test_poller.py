"""Tests for LiveGamePoller using recorded fixtures."""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlb_pipeline.config import Settings
from mlb_pipeline.ingestion.poller import LiveGamePoller, TrackedGame
from mlb_pipeline.models.enums import GameState

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
def test_settings():
    return Settings(
        db_server="localhost",
        db_name="test_db",
        poll_interval_live=0.01,
        poll_interval_idle=0.01,
        poll_interval_pregame=0.01,
    )


class TestScheduleDiscovery:
    @pytest.mark.asyncio
    async def test_discovers_games_from_schedule(
        self, mock_client, mock_storage, test_settings, schedule_data
    ):
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        await poller._poll_schedule(date(2026, 3, 28))
        # Schedule fixture has 15 games
        game_count = len(schedule_data["dates"][0]["games"])
        assert len(poller._games) == game_count
        assert poller._stats.games_discovered == game_count

    @pytest.mark.asyncio
    async def test_upserts_game_to_storage(
        self, mock_client, mock_storage, test_settings
    ):
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        await poller._poll_schedule(date(2026, 3, 28))
        assert mock_storage.upsert_game.call_count > 0

    @pytest.mark.asyncio
    async def test_detects_state_change(
        self, mock_client, mock_storage, test_settings, schedule_data
    ):
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        # First poll
        await poller._poll_schedule(date(2026, 3, 28))
        first_game_pk = schedule_data["dates"][0]["games"][0]["gamePk"]
        tracked = poller._games[first_game_pk]
        original_state = tracked.state

        # Modify fixture to simulate state change
        modified_schedule = json.loads(json.dumps(schedule_data))
        modified_schedule["dates"][0]["games"][0]["status"]["abstractGameState"] = "Live"
        mock_client.get_schedule = AsyncMock(return_value=modified_schedule)

        await poller._poll_schedule(date(2026, 3, 28))
        if original_state != GameState.LIVE:
            assert tracked.state == GameState.LIVE


class TestDiffDetection:
    @pytest.mark.asyncio
    async def test_processes_all_plays_from_scratch(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """From last_completed_play_index=-1, all completed plays are new."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243,
            state=GameState.LIVE,
            away_team="New York Yankees",
            home_team="San Francisco Giants",
        )
        poller._games[823243] = game

        # Set return value to count pitches
        pitch_counts = []
        ab_counts = []

        def track_pitches(pitches):
            pitch_counts.append(len(pitches))
            return len(pitches)

        def track_at_bats(at_bats):
            ab_counts.append(len(at_bats))
            return len(at_bats)

        mock_storage.insert_pitches_batch = MagicMock(side_effect=track_pitches)
        mock_storage.insert_at_bats_batch = MagicMock(side_effect=track_at_bats)

        await poller._poll_live_game(game)

        total_pitches = sum(pitch_counts)
        total_at_bats = sum(ab_counts)
        assert total_pitches == 275
        assert total_at_bats == 67

    @pytest.mark.asyncio
    async def test_no_new_plays_when_caught_up(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """When already caught up, no new events stored."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        all_plays = live_feed_data["liveData"]["plays"]["allPlays"]
        game = TrackedGame(
            game_pk=823243,
            state=GameState.LIVE,
            last_completed_play_index=len(all_plays) - 1,
            away_team="New York Yankees",
            home_team="San Francisco Giants",
        )
        poller._games[823243] = game

        await poller._poll_live_game(game)

        # insert_pitches_batch should not have been called, or called with empty
        if mock_storage.insert_pitches_batch.called:
            args = mock_storage.insert_pitches_batch.call_args
            assert len(args[0][0]) == 0 or not args

    @pytest.mark.asyncio
    async def test_incremental_processing(
        self, mock_client, mock_storage, test_settings, live_feed_data
    ):
        """After processing some plays, only new ones are handled."""
        poller = LiveGamePoller(
            client=mock_client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(
            game_pk=823243,
            state=GameState.LIVE,
            last_completed_play_index=9,  # Already processed first 10 plays
            away_team="New York Yankees",
            home_team="San Francisco Giants",
        )
        poller._games[823243] = game

        pitch_counts = []
        mock_storage.insert_pitches_batch = MagicMock(
            side_effect=lambda p: (pitch_counts.append(len(p)), len(p))[1]
        )

        await poller._poll_live_game(game)

        total = sum(pitch_counts)
        # Should be less than 275 since first 10 plays already processed
        assert total < 275
        assert total > 0


class TestPollInterval:
    def test_live_interval(self, test_settings):
        poller = LiveGamePoller(
            client=AsyncMock(),
            storage=MagicMock(),
            publisher=None,
            settings=test_settings,
        )
        poller._games[1] = TrackedGame(game_pk=1, state=GameState.LIVE)
        assert poller._determine_poll_interval() == test_settings.poll_interval_live

    def test_pregame_interval(self, test_settings):
        poller = LiveGamePoller(
            client=AsyncMock(),
            storage=MagicMock(),
            publisher=None,
            settings=test_settings,
        )
        poller._games[1] = TrackedGame(game_pk=1, state=GameState.PREVIEW)
        assert poller._determine_poll_interval() == test_settings.poll_interval_pregame

    def test_idle_interval(self, test_settings):
        poller = LiveGamePoller(
            client=AsyncMock(),
            storage=MagicMock(),
            publisher=None,
            settings=test_settings,
        )
        poller._games[1] = TrackedGame(game_pk=1, state=GameState.FINAL)
        assert poller._determine_poll_interval() == test_settings.poll_interval_idle


class TestErrorResilience:
    @pytest.mark.asyncio
    async def test_api_failure_continues(self, mock_storage, test_settings):
        """If get_live_feed raises, the poller logs and continues."""
        client = AsyncMock()
        client.get_live_feed = AsyncMock(side_effect=Exception("API down"))

        poller = LiveGamePoller(
            client=client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        game = TrackedGame(game_pk=1, state=GameState.LIVE)
        poller._games[1] = game

        # Should not raise
        await poller._poll_live_game(game)
        assert game.errors == 1
        assert poller._stats.errors == 1

    @pytest.mark.asyncio
    async def test_schedule_failure_continues(self, mock_storage, test_settings):
        """If get_schedule raises, poller logs and continues."""
        client = AsyncMock()
        client.get_schedule = AsyncMock(side_effect=Exception("Network error"))

        poller = LiveGamePoller(
            client=client,
            storage=mock_storage,
            publisher=None,
            settings=test_settings,
        )
        await poller._poll_schedule(date(2026, 3, 28))
        assert poller._stats.errors == 1
        assert len(poller._games) == 0
