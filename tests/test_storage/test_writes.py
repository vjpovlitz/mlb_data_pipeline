"""Tests for SQLServerStorage write operations."""

import json
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from mlb_pipeline.config import settings
from mlb_pipeline.ingestion.parser import parse_all_game_events
from mlb_pipeline.models.database import AtBat, Game, Pitch
from mlb_pipeline.storage.sqlserver import SQLServerStorage

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TEST_GAME_PK = 823243


@pytest.fixture
def storage():
    engine = create_engine(settings.db_connection_string)
    return SQLServerStorage(engine)


@pytest.fixture
def live_feed_data():
    with open(FIXTURES_DIR / "live_feed_sample.json") as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def cleanup(storage):
    """Clean up test data after each test."""
    yield
    with Session(storage._engine) as session:
        session.query(Pitch).filter(Pitch.game_pk == TEST_GAME_PK).delete()
        session.query(AtBat).filter(AtBat.game_pk == TEST_GAME_PK).delete()
        session.query(Game).filter(Game.game_pk == TEST_GAME_PK).delete()
        session.commit()


class TestUpsertGame:
    def test_insert_new_game(self, storage):
        storage.upsert_game(
            game_pk=TEST_GAME_PK,
            game_date=date(2026, 3, 27),
            game_type="R",
            season=2026,
            status="Preview",
            away_team_id=147,
            away_team_name="New York Yankees",
            home_team_id=137,
            home_team_name="San Francisco Giants",
            venue_name="Oracle Park",
        )
        game = storage.get_game(TEST_GAME_PK)
        assert game is not None
        assert game.status == "Preview"
        assert game.away_score == 0

    def test_update_existing_game_score(self, storage):
        storage.upsert_game(
            game_pk=TEST_GAME_PK,
            game_date=date(2026, 3, 27),
            game_type="R",
            season=2026,
            status="Preview",
            away_team_id=147,
            away_team_name="New York Yankees",
            home_team_id=137,
            home_team_name="San Francisco Giants",
        )
        storage.update_game_score(TEST_GAME_PK, away_score=3, home_score=0, status="Final")
        game = storage.get_game(TEST_GAME_PK)
        assert game is not None
        assert game.away_score == 3
        assert game.home_score == 0
        assert game.status == "Final"
        assert game.home_team_won is False

    def test_upsert_idempotent(self, storage):
        for _ in range(3):
            storage.upsert_game(
                game_pk=TEST_GAME_PK,
                game_date=date(2026, 3, 27),
                game_type="R",
                season=2026,
                status="Live",
                away_team_id=147,
                away_team_name="New York Yankees",
                home_team_id=137,
                home_team_name="San Francisco Giants",
            )
        with Session(storage._engine) as session:
            count = session.query(Game).filter(Game.game_pk == TEST_GAME_PK).count()
            assert count == 1


class TestInsertPitches:
    def test_insert_batch_from_fixture(self, storage, live_feed_data):
        pitches, _ = parse_all_game_events(TEST_GAME_PK, live_feed_data)
        count = storage.insert_pitches_batch(pitches)
        assert count == 275
        with Session(storage._engine) as session:
            db_count = session.query(Pitch).filter(Pitch.game_pk == TEST_GAME_PK).count()
            assert db_count == 275

    def test_insert_idempotent_no_duplicates(self, storage, live_feed_data):
        pitches, _ = parse_all_game_events(TEST_GAME_PK, live_feed_data)
        storage.insert_pitches_batch(pitches)
        storage.insert_pitches_batch(pitches)  # second insert
        with Session(storage._engine) as session:
            db_count = session.query(Pitch).filter(Pitch.game_pk == TEST_GAME_PK).count()
            assert db_count == 275

    def test_empty_batch(self, storage):
        count = storage.insert_pitches_batch([])
        assert count == 0


class TestInsertAtBats:
    def test_insert_batch_from_fixture(self, storage, live_feed_data):
        _, at_bats = parse_all_game_events(TEST_GAME_PK, live_feed_data)
        count = storage.insert_at_bats_batch(at_bats)
        assert count == 67
        with Session(storage._engine) as session:
            db_count = session.query(AtBat).filter(AtBat.game_pk == TEST_GAME_PK).count()
            assert db_count == 67

    def test_insert_idempotent(self, storage, live_feed_data):
        _, at_bats = parse_all_game_events(TEST_GAME_PK, live_feed_data)
        storage.insert_at_bats_batch(at_bats)
        storage.insert_at_bats_batch(at_bats)
        with Session(storage._engine) as session:
            db_count = session.query(AtBat).filter(AtBat.game_pk == TEST_GAME_PK).count()
            assert db_count == 67


class TestFullGameInsert:
    def test_insert_entire_game_from_fixture(self, storage, live_feed_data):
        """Parse and store all events from the fixture. Verify counts."""
        pitches, at_bats = parse_all_game_events(TEST_GAME_PK, live_feed_data)

        storage.upsert_game(
            game_pk=TEST_GAME_PK,
            game_date=date(2026, 3, 27),
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

        pitches_inserted = storage.insert_pitches_batch(pitches)
        at_bats_inserted = storage.insert_at_bats_batch(at_bats)

        assert pitches_inserted == 275
        assert at_bats_inserted == 67

        game = storage.get_game(TEST_GAME_PK)
        assert game is not None
        assert game.home_team_won is False
