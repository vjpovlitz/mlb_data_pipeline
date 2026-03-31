"""Tests for DuckDBStore — analytical queries over Parquet files."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import create_engine

from mlb_pipeline.config import settings
from mlb_pipeline.ingestion.parser import parse_all_game_events
from mlb_pipeline.storage.duckdb_store import DuckDBStore
from mlb_pipeline.storage.parquet import ParquetStore
from mlb_pipeline.storage.sqlserver import SQLServerStorage

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_GAME_PK = 823243
TEST_GAME_DATE = date(2026, 3, 27)


@pytest.fixture
def parquet_dir(tmp_path):
    return tmp_path


@pytest.fixture
def populated_parquet(parquet_dir):
    """Export fixture game data to Parquet for DuckDB to query."""
    engine = create_engine(settings.db_connection_string)
    storage = SQLServerStorage(engine)

    with open(FIXTURES_DIR / "live_feed_sample.json") as f:
        feed = json.load(f)

    # Ensure game is in SQL Server
    storage.upsert_game(
        game_pk=TEST_GAME_PK,
        game_date=TEST_GAME_DATE,
        game_type="R",
        season=2026,
        status="Final",
        away_team_id=147,
        away_team_name="New York Yankees",
        home_team_id=137,
        home_team_name="San Francisco Giants",
        venue_name="Oracle Park",
        away_score=3,
        home_score=0,
    )
    pitches, at_bats = parse_all_game_events(TEST_GAME_PK, feed)
    storage.insert_pitches_batch(pitches)
    storage.insert_at_bats_batch(at_bats)

    pq_store = ParquetStore(parquet_dir)
    pq_store.export_date_range(engine, TEST_GAME_DATE, TEST_GAME_DATE)
    return parquet_dir


@pytest.fixture
def duckdb_store(populated_parquet):
    store = DuckDBStore(populated_parquet)
    yield store
    store.close()


class TestDuckDBStoreInit:
    def test_creates_connection(self, parquet_dir):
        store = DuckDBStore(parquet_dir)
        assert store._conn is not None
        store.close()


class TestPitcherArsenal:
    def test_returns_pitch_types(self, duckdb_store):
        # Get a pitcher_id from the fixture
        pitches = pl.read_parquet(
            str(duckdb_store._base / "pitches" / str(TEST_GAME_DATE) / f"{TEST_GAME_PK}.parquet")
        )
        pitcher_id = pitches["pitcher_id"][0]

        result = duckdb_store.pitcher_arsenal(pitcher_id)
        assert result.height > 0
        assert "pitch_type" in result.columns
        assert "avg_speed" in result.columns
        assert "whiff_rate" in result.columns

    def test_nonexistent_pitcher_returns_empty(self, duckdb_store):
        result = duckdb_store.pitcher_arsenal(999999)
        assert result.is_empty()


class TestMatchupHistory:
    def test_returns_matchup_stats(self, duckdb_store):
        pitches = pl.read_parquet(
            str(duckdb_store._base / "pitches" / str(TEST_GAME_DATE) / f"{TEST_GAME_PK}.parquet")
        )
        pitcher_id = pitches["pitcher_id"][0]
        batter_id = pitches["batter_id"][0]

        result = duckdb_store.matchup_history(pitcher_id, batter_id)
        assert result.height > 0
        assert result["total_pitches"][0] > 0


class TestTeamStandings:
    def test_returns_standings(self, duckdb_store):
        result = duckdb_store.team_standings(2026)
        assert result.height >= 2  # At least NYY and SF
        assert "win_pct" in result.columns
        assert "run_diff" in result.columns


class TestDailyPitchLeaders:
    def test_returns_leaders(self, duckdb_store):
        result = duckdb_store.daily_pitch_leaders(str(TEST_GAME_DATE))
        assert result.height > 0
        assert "strikeouts" in result.columns
        assert "avg_speed" in result.columns


class TestGamePace:
    def test_returns_pace_data(self, duckdb_store):
        result = duckdb_store.game_pace(TEST_GAME_PK)
        assert result.height > 0
        assert "pitch_count" in result.columns
        assert "inning" in result.columns


class TestEmptyParquet:
    def test_queries_return_empty_when_no_data(self, parquet_dir):
        store = DuckDBStore(parquet_dir)
        assert store.pitcher_arsenal(1).is_empty()
        assert store.team_standings(2099).is_empty()
        assert store.daily_pitch_leaders("2099-01-01").is_empty()
        store.close()
