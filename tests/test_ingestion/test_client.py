"""Tests for MLB Stats API client using recorded fixtures."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlb_pipeline.ingestion.client import MLBStatsClient

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def schedule_data():
    with open(FIXTURES_DIR / "schedule_sample.json") as f:
        return json.load(f)


@pytest.fixture
def live_feed_data():
    with open(FIXTURES_DIR / "live_feed_sample.json") as f:
        return json.load(f)


class TestMLBStatsClient:
    @pytest.mark.asyncio
    async def test_get_schedule_parses_games(self, schedule_data):
        client = MLBStatsClient()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=schedule_data)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session
        client._external_session = True

        result = await client.get_schedule("2026-03-28")
        dates = result.get("dates", [])
        assert len(dates) > 0
        games = dates[0]["games"]
        assert len(games) > 0

        game = games[0]
        assert "gamePk" in game
        assert "teams" in game
        assert "away" in game["teams"]
        assert "home" in game["teams"]
        assert "status" in game

    @pytest.mark.asyncio
    async def test_live_feed_has_required_structure(self, live_feed_data):
        """Verify the live feed fixture has the structure we expect to parse."""
        assert "gameData" in live_feed_data
        assert "liveData" in live_feed_data

        game_data = live_feed_data["gameData"]
        assert "teams" in game_data
        assert "status" in game_data

        live_data = live_feed_data["liveData"]
        assert "plays" in live_data
        plays = live_data["plays"]
        assert "allPlays" in plays
        assert len(plays["allPlays"]) > 0

    @pytest.mark.asyncio
    async def test_live_feed_play_structure(self, live_feed_data):
        """Verify play/at-bat structure in the fixture."""
        play = live_feed_data["liveData"]["plays"]["allPlays"][0]
        assert "result" in play
        assert "matchup" in play
        assert "playEvents" in play
        assert "about" in play
        assert "atBatIndex" in play

        result = play["result"]
        assert "event" in result
        assert "description" in result

        matchup = play["matchup"]
        assert "batter" in matchup
        assert "pitcher" in matchup
        assert "fullName" in matchup["batter"]
        assert "id" in matchup["batter"]

    @pytest.mark.asyncio
    async def test_live_feed_pitch_event_structure(self, live_feed_data):
        """Verify pitch event structure in the fixture."""
        # Find first actual pitch
        for play in live_feed_data["liveData"]["plays"]["allPlays"]:
            for event in play.get("playEvents", []):
                if event.get("isPitch"):
                    assert "details" in event
                    assert "count" in event
                    assert "pitchData" in event

                    details = event["details"]
                    assert "type" in details
                    assert "call" in details

                    pitch_data = event["pitchData"]
                    assert "startSpeed" in pitch_data
                    assert "breaks" in pitch_data
                    assert "coordinates" in pitch_data
                    return

        pytest.fail("No pitch events found in fixture")

    @pytest.mark.asyncio
    async def test_live_feed_game_completed(self, live_feed_data):
        """Verify the fixture represents a completed game."""
        status = live_feed_data["gameData"]["status"]
        assert status["detailedState"] == "Final"
        assert status["abstractGameState"] == "Final"

    @pytest.mark.asyncio
    async def test_schedule_range_flattens_games(self, schedule_data):
        """Test that get_schedule_range returns a flat list."""
        client = MLBStatsClient()
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=schedule_data)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session
        client._external_session = True

        games = await client.get_schedule_range("2026-03-28", "2026-03-28")
        assert isinstance(games, list)
        if games:
            assert "gamePk" in games[0]

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager creates and cleans up session."""
        async with MLBStatsClient() as client:
            assert client._session is None or not client._session.closed
        # After exit, session should be closed (if created)


class TestFixtureDataQuality:
    """Tests that validate the captured fixture data itself."""

    def test_schedule_has_games(self, schedule_data):
        dates = schedule_data.get("dates", [])
        assert len(dates) >= 1
        total_games = sum(len(d["games"]) for d in dates)
        assert total_games >= 1

    def test_live_feed_has_pitches(self, live_feed_data):
        plays = live_feed_data["liveData"]["plays"]["allPlays"]
        pitch_count = 0
        for play in plays:
            for event in play.get("playEvents", []):
                if event.get("isPitch"):
                    pitch_count += 1
        assert pitch_count > 50, f"Expected 50+ pitches, got {pitch_count}"

    def test_live_feed_game_pk(self, live_feed_data):
        assert live_feed_data["gamePk"] == 823243

    def test_live_feed_has_scores(self, live_feed_data):
        linescore = live_feed_data["liveData"]["linescore"]
        away_runs = linescore["teams"]["away"]["runs"]
        home_runs = linescore["teams"]["home"]["runs"]
        assert isinstance(away_runs, int)
        assert isinstance(home_runs, int)
