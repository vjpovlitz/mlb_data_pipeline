"""Tests for ParquetStore — export, read, and compact operations."""

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import create_engine

from mlb_pipeline.config import settings
from mlb_pipeline.ingestion.parser import parse_all_game_events
from mlb_pipeline.storage.parquet import ParquetStore
from mlb_pipeline.storage.sqlserver import SQLServerStorage

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_GAME_PK = 823243
TEST_GAME_DATE = date(2026, 3, 27)


@pytest.fixture
def engine():
    return create_engine(settings.db_connection_string)


@pytest.fixture
def storage(engine):
    return SQLServerStorage(engine)


@pytest.fixture
def parquet_store(tmp_path):
    return ParquetStore(tmp_path)


@pytest.fixture
def live_feed_data():
    with open(FIXTURES_DIR / "live_feed_sample.json") as f:
        return json.load(f)


@pytest.fixture
def seeded_game(storage, live_feed_data):
    """Ensure the test game exists in SQL Server for export tests."""
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
    pitches, at_bats = parse_all_game_events(TEST_GAME_PK, live_feed_data)
    storage.insert_pitches_batch(pitches)
    storage.insert_at_bats_batch(at_bats)
    yield
    # Cleanup handled by test_writes cleanup fixture or just leave it


class TestExportGamePitches:
    def test_exports_pitches_to_parquet(self, parquet_store, engine, seeded_game):
        path = parquet_store.export_game_pitches(engine, TEST_GAME_PK)
        assert path.exists()
        df = pl.read_parquet(path)
        assert df.height == 275
        assert "game_pk" in df.columns

    def test_partitioned_by_date(self, parquet_store, engine, seeded_game):
        path = parquet_store.export_game_pitches(engine, TEST_GAME_PK)
        assert str(TEST_GAME_DATE) in str(path)


class TestExportGameAtBats:
    def test_exports_at_bats_to_parquet(self, parquet_store, engine, seeded_game):
        path = parquet_store.export_game_at_bats(engine, TEST_GAME_PK)
        assert path.exists()
        df = pl.read_parquet(path)
        assert df.height == 67


class TestExportGames:
    def test_exports_games(self, parquet_store, engine, seeded_game):
        path = parquet_store.export_games(engine, TEST_GAME_DATE, TEST_GAME_DATE)
        assert path.exists()
        df = pl.read_parquet(path)
        assert df.height >= 1
        assert TEST_GAME_PK in df["game_pk"].to_list()


class TestExportDateRange:
    def test_exports_all_data(self, parquet_store, engine, seeded_game):
        result = parquet_store.export_date_range(engine, TEST_GAME_DATE, TEST_GAME_DATE)
        assert result["games"] >= 1
        assert result["pitches"] == 275
        assert result["at_bats"] == 67


class TestReadPitches:
    def test_read_by_game_pk(self, parquet_store, engine, seeded_game):
        parquet_store.export_game_pitches(engine, TEST_GAME_PK)
        df = parquet_store.read_pitches(game_pk=TEST_GAME_PK)
        assert df.height == 275

    def test_read_by_date(self, parquet_store, engine, seeded_game):
        parquet_store.export_game_pitches(engine, TEST_GAME_PK)
        df = parquet_store.read_pitches(game_date=TEST_GAME_DATE)
        assert df.height == 275

    def test_read_nonexistent_returns_empty(self, parquet_store):
        df = parquet_store.read_pitches(game_pk=999999)
        assert df.is_empty()


class TestReadAtBats:
    def test_read_by_game_pk(self, parquet_store, engine, seeded_game):
        parquet_store.export_game_at_bats(engine, TEST_GAME_PK)
        df = parquet_store.read_at_bats(game_pk=TEST_GAME_PK)
        assert df.height == 67


class TestCompactPartitions:
    def test_compacts_monthly(self, parquet_store, engine, seeded_game):
        parquet_store.export_game_pitches(engine, TEST_GAME_PK)
        path = parquet_store.compact_date_partitions("pitches", 2026, 3)
        assert path.exists()
        df = pl.read_parquet(path)
        assert df.height == 275

    def test_compact_empty_returns_empty_path(self, parquet_store):
        path = parquet_store.compact_date_partitions("pitches", 2099, 1)
        assert path == Path()
