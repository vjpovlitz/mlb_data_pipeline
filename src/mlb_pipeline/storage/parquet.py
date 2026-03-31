"""Parquet export and partitioning layer.

Reads data from SQL Server via SQLAlchemy and writes partitioned Parquet
files using Polars for downstream analytics and model training.
"""

from datetime import date
from pathlib import Path

import polars as pl
import structlog
from sqlalchemy.engine import Engine

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Dataset subdirectory names
# ---------------------------------------------------------------------------
_DATASETS = ("pitches", "at_bats", "games", "daily_stats")


class ParquetStore:
    """Manage partitioned Parquet files for the MLB data pipeline."""

    def __init__(self, base_dir: Path) -> None:
        self._base = Path(base_dir)
        for name in _DATASETS:
            (self._base / name).mkdir(parents=True, exist_ok=True)
        logger.info("parquet_store_init", base_dir=str(self._base))

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_game_pitches(self, engine: Engine, game_pk: int) -> Path:
        """Export all pitches for a single game to a date-partitioned Parquet file.

        Layout: ``pitches/{game_date}/{game_pk}.parquet``
        """
        query = (
            f"SELECT p.*, g.game_date "
            f"FROM pitches p JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE p.game_pk = {int(game_pk)} ORDER BY p.at_bat_index, p.pitch_number"
        )
        with engine.connect() as conn:
            df = pl.read_database(query, connection=conn, schema_overrides={"game_pk": pl.Int64})

        if df.is_empty():
            logger.warning("no_pitches_found", game_pk=game_pk)
            return Path()

        game_date_str = str(df["game_date"][0])
        out_dir = self._base / "pitches" / game_date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{game_pk}.parquet"
        df.drop("game_date").write_parquet(out_path)
        logger.info("exported_pitches", game_pk=game_pk, rows=len(df), path=str(out_path))
        return out_path

    def export_game_at_bats(self, engine: Engine, game_pk: int) -> Path:
        """Export all at-bats for a single game to a date-partitioned Parquet file.

        Layout: ``at_bats/{game_date}/{game_pk}.parquet``
        """
        query = (
            f"SELECT ab.*, g.game_date "
            f"FROM at_bats ab JOIN games g ON ab.game_pk = g.game_pk "
            f"WHERE ab.game_pk = {int(game_pk)} ORDER BY ab.at_bat_index"
        )
        with engine.connect() as conn:
            df = pl.read_database(query, connection=conn, schema_overrides={"game_pk": pl.Int64})

        if df.is_empty():
            logger.warning("no_at_bats_found", game_pk=game_pk)
            return Path()

        game_date_str = str(df["game_date"][0])
        out_dir = self._base / "at_bats" / game_date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{game_pk}.parquet"
        df.drop("game_date").write_parquet(out_path)
        logger.info("exported_at_bats", game_pk=game_pk, rows=len(df), path=str(out_path))
        return out_path

    def export_games(self, engine: Engine, start_date: date, end_date: date) -> Path:
        """Export the games table for a date range.

        Layout: ``games/{season}/games.parquet``
        """
        query = (
            f"SELECT * FROM games "
            f"WHERE game_date >= '{start_date}' AND game_date <= '{end_date}' "
            f"ORDER BY game_date, game_pk"
        )
        with engine.connect() as conn:
            df = pl.read_database(
                query,
                connection=conn,
                schema_overrides={"game_pk": pl.Int64},
            )

        if df.is_empty():
            logger.warning("no_games_found", start_date=str(start_date), end_date=str(end_date))
            return Path()

        season = int(df["season"][0])
        out_dir = self._base / "games" / str(season)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "games.parquet"
        df.write_parquet(out_path)
        logger.info(
            "exported_games",
            rows=len(df),
            start_date=str(start_date),
            end_date=str(end_date),
            path=str(out_path),
        )
        return out_path

    def export_date_range(
        self,
        engine: Engine,
        start_date: date,
        end_date: date,
    ) -> dict[str, int]:
        """Export all games, pitches, and at-bats for a date range.

        Each game's pitches and at-bats are partitioned by date.  Returns a
        summary dict with counts, e.g. ``{"games": 5, "pitches": 1200, "at_bats": 300}``.
        """
        # Fetch game PKs in range
        game_query = (
            f"SELECT game_pk FROM games "
            f"WHERE game_date >= '{start_date}' AND game_date <= '{end_date}' "
            f"ORDER BY game_date"
        )
        with engine.connect() as conn:
            games_df = pl.read_database(
                game_query,
                connection=conn,
                schema_overrides={"game_pk": pl.Int64},
            )

        game_pks = games_df["game_pk"].to_list() if not games_df.is_empty() else []

        counts: dict[str, int] = {"games": 0, "pitches": 0, "at_bats": 0}

        # Export the games table itself
        if game_pks:
            self.export_games(engine, start_date, end_date)
            counts["games"] = len(game_pks)

        # Export pitches and at-bats per game
        for gpk in game_pks:
            pitch_path = self.export_game_pitches(engine, gpk)
            if pitch_path != Path():
                pitch_df = pl.read_parquet(pitch_path)
                counts["pitches"] += len(pitch_df)

            ab_path = self.export_game_at_bats(engine, gpk)
            if ab_path != Path():
                ab_df = pl.read_parquet(ab_path)
                counts["at_bats"] += len(ab_df)

        logger.info(
            "export_date_range_complete",
            start_date=str(start_date),
            end_date=str(end_date),
            **counts,
        )
        return counts

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def read_pitches(
        self,
        game_pk: int | None = None,
        game_date: date | None = None,
    ) -> pl.DataFrame:
        """Read pitches from Parquet files.

        Provide *game_pk* (scans all date dirs) or *game_date* (reads entire
        date partition) or both for a precise lookup.
        """
        pitches_dir = self._base / "pitches"

        if game_pk is not None and game_date is not None:
            target = pitches_dir / str(game_date) / f"{game_pk}.parquet"
            if not target.exists():
                logger.warning("parquet_not_found", path=str(target))
                return pl.DataFrame()
            return pl.read_parquet(target)

        if game_date is not None:
            date_dir = pitches_dir / str(game_date)
            if not date_dir.exists():
                logger.warning("date_dir_not_found", path=str(date_dir))
                return pl.DataFrame()
            files = sorted(date_dir.glob("*.parquet"))
            if not files:
                return pl.DataFrame()
            return pl.concat([pl.read_parquet(f) for f in files])

        if game_pk is not None:
            # Scan all date directories for the game
            frames: list[pl.DataFrame] = []
            for parquet_file in sorted(pitches_dir.rglob(f"{game_pk}.parquet")):
                frames.append(pl.read_parquet(parquet_file))
            if not frames:
                logger.warning("parquet_not_found", game_pk=game_pk)
                return pl.DataFrame()
            return pl.concat(frames)

        # No filter — read everything
        files = sorted(pitches_dir.rglob("*.parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in files])

    def read_at_bats(
        self,
        game_pk: int | None = None,
        game_date: date | None = None,
    ) -> pl.DataFrame:
        """Read at-bats from Parquet files.

        Provide *game_pk* (scans all date dirs) or *game_date* (reads entire
        date partition) or both for a precise lookup.
        """
        ab_dir = self._base / "at_bats"

        if game_pk is not None and game_date is not None:
            target = ab_dir / str(game_date) / f"{game_pk}.parquet"
            if not target.exists():
                logger.warning("parquet_not_found", path=str(target))
                return pl.DataFrame()
            return pl.read_parquet(target)

        if game_date is not None:
            date_dir = ab_dir / str(game_date)
            if not date_dir.exists():
                logger.warning("date_dir_not_found", path=str(date_dir))
                return pl.DataFrame()
            files = sorted(date_dir.glob("*.parquet"))
            if not files:
                return pl.DataFrame()
            return pl.concat([pl.read_parquet(f) for f in files])

        if game_pk is not None:
            frames: list[pl.DataFrame] = []
            for parquet_file in sorted(ab_dir.rglob(f"{game_pk}.parquet")):
                frames.append(pl.read_parquet(parquet_file))
            if not frames:
                logger.warning("parquet_not_found", game_pk=game_pk)
                return pl.DataFrame()
            return pl.concat(frames)

        # No filter — read everything
        files = sorted(ab_dir.rglob("*.parquet"))
        if not files:
            return pl.DataFrame()
        return pl.concat([pl.read_parquet(f) for f in files])

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact_date_partitions(self, dataset: str, year: int, month: int) -> Path:
        """Merge all daily Parquet files for a given month into one file.

        This improves read performance for queries spanning an entire month.
        The merged file is written to ``{dataset}/{year}-{month:02d}.parquet``.
        """
        ds_dir = self._base / dataset
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {ds_dir}")

        # Collect all parquet files whose parent dir falls within the month
        frames: list[pl.DataFrame] = []
        for date_dir in sorted(ds_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            try:
                dir_date = date.fromisoformat(date_dir.name)
            except ValueError:
                continue
            if dir_date.year == year and dir_date.month == month:
                for pf in sorted(date_dir.glob("*.parquet")):
                    frames.append(pl.read_parquet(pf))

        if not frames:
            logger.warning(
                "no_partitions_to_compact",
                dataset=dataset,
                year=year,
                month=month,
            )
            return Path()

        merged = pl.concat(frames)
        out_path = ds_dir / f"{year}-{month:02d}.parquet"
        merged.write_parquet(out_path)
        logger.info(
            "compacted_partitions",
            dataset=dataset,
            year=year,
            month=month,
            rows=len(merged),
            source_files=len(frames),
            path=str(out_path),
        )
        return out_path
