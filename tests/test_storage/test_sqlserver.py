"""Tests for SQL Server connectivity and ORM round-trips."""

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from mlb_pipeline.config import settings
from mlb_pipeline.models.database import Base, Game, Team


@pytest.fixture
def engine():
    return create_engine(settings.db_connection_string)


class TestSQLServerConnection:
    def test_can_connect(self, engine):
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1

    def test_tables_exist(self, engine):
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES ORDER BY TABLE_NAME")
            ).fetchall()
            table_names = [r[0] for r in result]
            assert "games" in table_names
            assert "pitches" in table_names
            assert "at_bats" in table_names
            assert "teams" in table_names
            assert "players" in table_names
            assert "player_stats_daily" in table_names
            assert "win_probability_log" in table_names

    def test_teams_seeded(self, engine):
        with Session(engine) as session:
            teams = session.query(Team).all()
            assert len(teams) == 30

    def test_game_write_read_delete(self, engine):
        """Full round-trip: insert a game, read it back, delete it."""
        test_game = Game(
            game_pk=999999,
            game_date=date(2025, 6, 15),
            game_type="R",
            season=2025,
            status="Final",
            away_team_id=147,
            away_team_name="New York Yankees",
            home_team_id=137,
            home_team_name="San Francisco Giants",
            away_score=3,
            home_score=0,
            home_team_won=False,
        )

        with Session(engine) as session:
            session.add(test_game)
            session.commit()

        # Read back
        with Session(engine) as session:
            game = session.get(Game, 999999)
            assert game is not None
            assert game.away_team_name == "New York Yankees"
            assert game.home_team_name == "San Francisco Giants"
            assert game.away_score == 3
            assert game.home_score == 0
            assert game.home_team_won is False

        # Cleanup
        with Session(engine) as session:
            game = session.get(Game, 999999)
            if game:
                session.delete(game)
                session.commit()

        # Verify deleted
        with Session(engine) as session:
            game = session.get(Game, 999999)
            assert game is None
