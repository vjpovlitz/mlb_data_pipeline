"""SQL Server storage layer for the pipeline."""

from datetime import date
from typing import Sequence

import structlog
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mlb_pipeline.models.database import AtBat, Game, Pitch
from mlb_pipeline.models.events import AtBatResult, PitchEvent

logger = structlog.get_logger()


class SQLServerStorage:
    """Write pipeline events to SQL Server. All methods are synchronous."""

    def __init__(self, engine: Engine):
        self._engine = engine

    @classmethod
    def from_connection_string(cls, connection_string: str) -> "SQLServerStorage":
        engine = create_engine(connection_string)
        return cls(engine)

    def upsert_game(
        self,
        game_pk: int,
        game_date: date,
        game_type: str,
        season: int,
        status: str,
        away_team_id: int,
        away_team_name: str,
        home_team_id: int,
        home_team_name: str,
        venue_name: str | None = None,
        away_score: int = 0,
        home_score: int = 0,
    ) -> None:
        """Insert or update a game record."""
        game = Game(
            game_pk=game_pk,
            game_date=game_date,
            game_type=game_type,
            season=season,
            status=status,
            away_team_id=away_team_id,
            away_team_name=away_team_name,
            home_team_id=home_team_id,
            home_team_name=home_team_name,
            venue_name=venue_name,
            away_score=away_score,
            home_score=home_score,
            home_team_won=home_score > away_score if status == "Final" else None,
        )
        with Session(self._engine) as session:
            session.merge(game)
            session.commit()

    def insert_pitches_batch(self, pitches: Sequence[PitchEvent]) -> int:
        """Insert pitches, skipping duplicates via merge. Returns count inserted."""
        if not pitches:
            return 0
        count = 0
        with Session(self._engine) as session:
            for p in pitches:
                data = p.model_dump()
                data["half_inning"] = p.half_inning.value
                ts = data["timestamp"]
                if ts.tzinfo is not None:
                    data["timestamp"] = ts.replace(tzinfo=None)
                pitch = Pitch(**data)
                session.merge(pitch)
                count += 1
            session.commit()
        logger.debug("pitches_stored", count=count)
        return count

    def insert_at_bats_batch(self, at_bats: Sequence[AtBatResult]) -> int:
        """Insert at-bats, skipping duplicates via merge. Returns count inserted."""
        if not at_bats:
            return 0
        count = 0
        with Session(self._engine) as session:
            for ab in at_bats:
                data = ab.model_dump()
                data["half_inning"] = ab.half_inning.value
                ts = data["timestamp"]
                if ts.tzinfo is not None:
                    data["timestamp"] = ts.replace(tzinfo=None)
                at_bat = AtBat(**data)
                session.merge(at_bat)
                count += 1
            session.commit()
        logger.debug("at_bats_stored", count=count)
        return count

    def update_game_score(
        self, game_pk: int, away_score: int, home_score: int, status: str
    ) -> None:
        """Update score and status for a game."""
        with Session(self._engine) as session:
            game = session.get(Game, game_pk)
            if game:
                game.away_score = away_score
                game.home_score = home_score
                game.status = status
                if status == "Final":
                    game.home_team_won = home_score > away_score
                session.commit()

    def get_game(self, game_pk: int) -> Game | None:
        """Retrieve a game record by primary key."""
        with Session(self._engine) as session:
            return session.get(Game, game_pk)
