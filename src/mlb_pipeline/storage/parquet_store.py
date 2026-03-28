"""Parquet file management with DuckDB analytical layer.

Games are stored as partitioned Parquet files:
  data/parquet/pitches/season=YYYY/game_pk=NNNN/pitches.parquet
  data/parquet/at_bats/season=YYYY/game_pk=NNNN/at_bats.parquet

DuckDB is used for fast analytical queries (feature engineering, model prep).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import structlog

from mlb_pipeline.config import settings
from mlb_pipeline.models.events import AtBatResult, PitchEvent

logger = structlog.get_logger(__name__)


class ParquetStore:
    """Read/write Parquet files and expose DuckDB for ad-hoc analytics."""

    def __init__(self, base_dir: Path | None = None):
        self._base = base_dir or settings.parquet_dir
        self._base.mkdir(parents=True, exist_ok=True)
        (self._base / "pitches").mkdir(exist_ok=True)
        (self._base / "at_bats").mkdir(exist_ok=True)
        (self._base / "win_prob_training").mkdir(exist_ok=True)
        self.log = logger.bind(component="parquet_store")

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def write_pitches(self, game_pk: int, season: int, events: list[PitchEvent]) -> Path:
        """Serialize pitches to Parquet, returns output path."""
        rows = [e.model_dump() for e in events]
        df = pl.DataFrame(rows)
        path = self._pitch_path(game_pk, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(path))
        self.log.info("pitches_written", game_pk=game_pk, rows=len(rows), path=str(path))
        return path

    def write_at_bats(self, game_pk: int, season: int, events: list[AtBatResult]) -> Path:
        """Serialize at-bats to Parquet, returns output path."""
        rows = [e.model_dump() for e in events]
        df = pl.DataFrame(rows)
        path = self._at_bat_path(game_pk, season)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(path))
        self.log.info("at_bats_written", game_pk=game_pk, rows=len(rows), path=str(path))
        return path

    def write_training_dataset(self, df: pl.DataFrame, name: str) -> Path:
        """Write a prepared training dataset."""
        path = self._base / "win_prob_training" / f"{name}.parquet"
        df.write_parquet(str(path))
        self.log.info("training_data_written", name=name, rows=len(df), path=str(path))
        return path

    # ------------------------------------------------------------------
    # DuckDB query interface
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pl.DataFrame:
        """Execute a DuckDB SQL query over Parquet files. Returns Polars DataFrame."""
        con = duckdb.connect(":memory:")
        # Register glob paths so queries can reference them by name
        pitches_glob = str(self._base / "pitches" / "**" / "*.parquet")
        at_bats_glob = str(self._base / "at_bats" / "**" / "*.parquet")
        con.execute(f"CREATE VIEW pitches AS SELECT * FROM read_parquet('{pitches_glob}', hive_partitioning=true)")
        con.execute(f"CREATE VIEW at_bats AS SELECT * FROM read_parquet('{at_bats_glob}', hive_partitioning=true)")
        result = con.execute(sql).pl()
        con.close()
        return result

    def load_pitches(self, game_pk: int | None = None, season: int | None = None) -> pl.DataFrame:
        """Load pitch data, optionally filtered by game or season."""
        clauses = []
        if game_pk is not None:
            clauses.append(f"game_pk = {game_pk}")
        if season is not None:
            clauses.append(f"season = {season}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT * FROM pitches {where}")

    def load_at_bats(self, game_pk: int | None = None, season: int | None = None) -> pl.DataFrame:
        """Load at-bat data, optionally filtered."""
        clauses = []
        if game_pk is not None:
            clauses.append(f"game_pk = {game_pk}")
        if season is not None:
            clauses.append(f"season = {season}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.query(f"SELECT * FROM at_bats {where}")

    def list_games(self) -> list[int]:
        """Return sorted list of all stored game_pks."""
        try:
            df = self.query("SELECT DISTINCT game_pk FROM at_bats ORDER BY game_pk")
            return df["game_pk"].to_list()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _pitch_path(self, game_pk: int, season: int) -> Path:
        return self._base / "pitches" / f"season={season}" / f"game_pk={game_pk}" / "pitches.parquet"

    def _at_bat_path(self, game_pk: int, season: int) -> Path:
        return self._base / "at_bats" / f"season={season}" / f"game_pk={game_pk}" / "at_bats.parquet"
