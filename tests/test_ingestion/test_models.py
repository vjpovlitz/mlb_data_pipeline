"""Tests for data models and configuration."""

from datetime import datetime

from mlb_pipeline.config import Settings
from mlb_pipeline.models.enums import GameState, HalfInning, PitchType
from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent


class TestSettings:
    def test_default_settings(self):
        s = Settings(
            db_server="localhost",
            db_name="mlb_pipeline",
            db_driver="ODBC Driver 17 for SQL Server",
        )
        assert s.db_server == "localhost"
        assert s.poll_interval_live == 2.0
        assert "mssql+pyodbc" in s.db_connection_string

    def test_connection_string_format(self):
        s = Settings(
            db_server="localhost",
            db_name="test_db",
            db_driver="ODBC Driver 17 for SQL Server",
        )
        assert "test_db" in s.db_connection_string
        assert "trusted_connection=yes" in s.db_connection_string


class TestEnums:
    def test_game_states(self):
        assert GameState.LIVE == "Live"
        assert GameState.FINAL == "Final"

    def test_half_inning(self):
        assert HalfInning.TOP == "top"
        assert HalfInning.BOTTOM == "bottom"

    def test_pitch_types(self):
        assert PitchType.FF == "FF"
        assert PitchType.SL == "SL"


class TestPitchEvent:
    def test_create_pitch_event(self):
        event = PitchEvent(
            game_pk=717465,
            event_id="717465_0_1",
            timestamp=datetime(2025, 6, 15, 19, 5, 0),
            inning=1,
            half_inning=HalfInning.TOP,
            at_bat_index=0,
            pitch_number=1,
            pitcher_id=543037,
            pitcher_name="Gerrit Cole",
            batter_id=665742,
            batter_name="Ronald Acuna Jr.",
            pitch_type="FF",
            start_speed=97.5,
            call_code="S",
            call_description="Called Strike",
            is_in_play=False,
            is_strike=True,
            is_ball=False,
            balls=0,
            strikes=1,
            outs=0,
        )
        assert event.game_pk == 717465
        assert event.is_strike is True
        assert event.launch_speed is None  # Not in play

    def test_pitch_event_json_round_trip(self):
        event = PitchEvent(
            game_pk=1,
            event_id="1_0_1",
            timestamp=datetime(2025, 1, 1),
            inning=1,
            half_inning=HalfInning.TOP,
            at_bat_index=0,
            pitch_number=1,
            pitcher_id=1,
            pitcher_name="Test Pitcher",
            batter_id=2,
            batter_name="Test Batter",
            call_code="B",
            call_description="Ball",
            is_in_play=False,
            is_strike=False,
            is_ball=True,
            balls=1,
            strikes=0,
            outs=0,
        )
        json_str = event.model_dump_json()
        restored = PitchEvent.model_validate_json(json_str)
        assert restored == event


class TestAtBatResult:
    def test_create_at_bat_result(self):
        result = AtBatResult(
            game_pk=717465,
            at_bat_index=0,
            timestamp=datetime(2025, 6, 15, 19, 8, 0),
            inning=1,
            half_inning=HalfInning.TOP,
            pitcher_id=543037,
            pitcher_name="Gerrit Cole",
            batter_id=665742,
            batter_name="Ronald Acuna Jr.",
            event="Strikeout",
            event_type="strikeout",
            description="Ronald Acuna Jr. strikes out swinging.",
            rbi=0,
            away_score=0,
            home_score=0,
            is_scoring_play=False,
            outs_after=1,
            pitch_count=6,
        )
        assert result.event == "Strikeout"
        assert result.is_scoring_play is False


class TestGameStateEvent:
    def test_create_game_state_event(self):
        event = GameStateEvent(
            game_pk=717465,
            timestamp=datetime(2025, 6, 15, 19, 5, 0),
            previous_state=GameState.PREVIEW,
            new_state=GameState.LIVE,
            away_team="Atlanta Braves",
            home_team="New York Yankees",
        )
        assert event.new_state == GameState.LIVE
