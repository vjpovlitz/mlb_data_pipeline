"""DuckDB analytical query layer for the MLB data pipeline.

Queries Parquet files on disk via DuckDB's ``read_parquet`` with glob patterns
and returns results as Polars DataFrames.  All paths use forward slashes so
that DuckDB handles them correctly on every platform.
"""

from pathlib import Path

import duckdb
import polars as pl
import structlog

logger = structlog.get_logger()


class DuckDBStore:
    """Analytical read layer backed by DuckDB over partitioned Parquet files.

    Expected directory layout under *parquet_dir*::

        pitches/{YYYY-MM-DD}/{game_pk}.parquet
        at_bats/{YYYY-MM-DD}/{game_pk}.parquet
        games/{season}/games.parquet
        daily_stats/{YYYY-MM-DD}.parquet
    """

    def __init__(self, parquet_dir: Path) -> None:
        self._base = Path(parquet_dir)
        self._conn = duckdb.connect(database=":memory:")
        logger.info("duckdb_store_init", parquet_dir=str(self._base))

    # ------------------------------------------------------------------
    # Glob helpers
    # ------------------------------------------------------------------

    def _pitches_glob(self, date_pattern: str = "*") -> str:
        """Return a glob pattern for pitch Parquet files."""
        return f"{self._base.as_posix()}/pitches/{date_pattern}/*.parquet"

    def _at_bats_glob(self, date_pattern: str = "*") -> str:
        """Return a glob pattern for at-bat Parquet files."""
        return f"{self._base.as_posix()}/at_bats/{date_pattern}/*.parquet"

    # ------------------------------------------------------------------
    # Analytical queries
    # ------------------------------------------------------------------

    def pitcher_arsenal(
        self,
        pitcher_id: int,
        season_dates: tuple[str, str] | None = None,
    ) -> pl.DataFrame:
        """Pitch-mix analysis for a single pitcher.

        Returns one row per pitch type with usage counts/percentages, average
        speed, spin rate, location, whiff rate, and zone rate.  Optionally
        filter by a ``(start_date, end_date)`` tuple (inclusive, ISO format).
        """
        date_pattern = "*"
        date_filter = ""
        if season_dates is not None:
            start, end = season_dates
            # Narrow the glob isn't practical for arbitrary ranges, so we
            # filter inside SQL instead.
            date_filter = f"AND game_date >= '{start}' AND game_date <= '{end}'"

        glob = self._pitches_glob(date_pattern)

        sql = f"""
            WITH base AS (
                SELECT
                    *,
                    -- Extract date from the file path (directory name)
                    regexp_extract(filename, '.*/pitches/([^/]+)/', 1) AS game_date
                FROM read_parquet('{glob}', filename=true)
                WHERE pitcher_id = {pitcher_id}
                {date_filter}
            ),
            total AS (
                SELECT COUNT(*) AS total_pitches FROM base
            )
            SELECT
                b.pitch_type,
                COUNT(*)                                            AS count,
                ROUND(COUNT(*) * 100.0 / t.total_pitches, 1)       AS pct,
                ROUND(AVG(b.start_speed), 1)                        AS avg_speed,
                ROUND(AVG(b.spin_rate), 0)                          AS avg_spin_rate,
                ROUND(AVG(b.plate_x), 3)                            AS avg_plate_x,
                ROUND(AVG(b.plate_z), 3)                            AS avg_plate_z,
                ROUND(
                    SUM(CASE WHEN b.call_code = 'S' AND b.is_strike AND NOT b.is_in_play
                             THEN 1 ELSE 0 END)
                    * 1.0 / COUNT(*), 3
                )                                                   AS whiff_rate,
                ROUND(
                    SUM(CASE WHEN b.zone IS NOT NULL AND b.zone BETWEEN 1 AND 9
                             THEN 1 ELSE 0 END)
                    * 1.0 / COUNT(*), 3
                )                                                   AS zone_rate
            FROM base b
            CROSS JOIN total t
            GROUP BY b.pitch_type, t.total_pitches
            ORDER BY count DESC
        """
        return self._execute(sql, "pitcher_arsenal", {
            "pitch_type": pl.Utf8,
            "count": pl.Int64,
            "pct": pl.Float64,
            "avg_speed": pl.Float64,
            "avg_spin_rate": pl.Float64,
            "avg_plate_x": pl.Float64,
            "avg_plate_z": pl.Float64,
            "whiff_rate": pl.Float64,
            "zone_rate": pl.Float64,
        })

    def matchup_history(
        self,
        pitcher_id: int,
        batter_id: int,
    ) -> pl.DataFrame:
        """Head-to-head historical stats between a pitcher and batter."""
        pitches_glob = self._pitches_glob()
        ab_glob = self._at_bats_glob()

        sql = f"""
            WITH ab AS (
                SELECT *
                FROM read_parquet('{ab_glob}')
                WHERE pitcher_id = {pitcher_id}
                  AND batter_id  = {batter_id}
            ),
            pitches AS (
                SELECT *
                FROM read_parquet('{pitches_glob}')
                WHERE pitcher_id = {pitcher_id}
                  AND batter_id  = {batter_id}
            ),
            pitch_types AS (
                SELECT STRING_AGG(DISTINCT pitch_type, ', ' ORDER BY pitch_type)
                    AS pitch_types_seen
                FROM pitches
                WHERE pitch_type IS NOT NULL
            )
            SELECT
                (SELECT COUNT(*) FROM pitches)          AS total_pitches,
                COUNT(*)                                AS total_at_bats,
                SUM(CASE WHEN ab.event_type IN (
                        'single', 'double', 'triple', 'home_run'
                    ) THEN 1 ELSE 0 END)                AS hits,
                SUM(CASE WHEN ab.event_type = 'strikeout'
                    THEN 1 ELSE 0 END)                  AS strikeouts,
                SUM(CASE WHEN ab.event_type = 'walk'
                    THEN 1 ELSE 0 END)                  AS walks,
                SUM(CASE WHEN ab.event_type = 'home_run'
                    THEN 1 ELSE 0 END)                  AS home_runs,
                ROUND(
                    SUM(CASE WHEN ab.event_type IN (
                            'single', 'double', 'triple', 'home_run'
                        ) THEN 1 ELSE 0 END)
                    * 1.0
                    / GREATEST(COUNT(*), 1), 3
                )                                       AS batting_avg,
                ANY_VALUE(pt.pitch_types_seen)           AS pitch_types_seen
            FROM ab
            CROSS JOIN pitch_types pt
        """
        return self._execute(sql, "matchup_history", {
            "total_pitches": pl.Int64,
            "total_at_bats": pl.Int64,
            "hits": pl.Int64,
            "strikeouts": pl.Int64,
            "walks": pl.Int64,
            "home_runs": pl.Int64,
            "batting_avg": pl.Float64,
            "pitch_types_seen": pl.Utf8,
        })

    def team_standings(self, season: int) -> pl.DataFrame:
        """Win/loss standings for every team in a season.

        Combines home and away records from the games Parquet file.
        """
        games_glob = f"{self._base.as_posix()}/games/{season}/games.parquet"

        sql = f"""
            WITH games AS (
                SELECT * FROM read_parquet('{games_glob}')
                WHERE status = 'Final'
            ),
            home AS (
                SELECT
                    home_team_name AS team_name,
                    SUM(CASE WHEN home_team_won THEN 1 ELSE 0 END)  AS wins,
                    SUM(CASE WHEN NOT home_team_won THEN 1 ELSE 0 END) AS losses,
                    SUM(home_score)  AS runs_scored,
                    SUM(away_score)  AS runs_allowed
                FROM games
                GROUP BY home_team_name
            ),
            away AS (
                SELECT
                    away_team_name AS team_name,
                    SUM(CASE WHEN NOT home_team_won THEN 1 ELSE 0 END)  AS wins,
                    SUM(CASE WHEN home_team_won THEN 1 ELSE 0 END) AS losses,
                    SUM(away_score)  AS runs_scored,
                    SUM(home_score)  AS runs_allowed
                FROM games
                GROUP BY away_team_name
            ),
            combined AS (
                SELECT
                    COALESCE(h.team_name, a.team_name)               AS team_name,
                    COALESCE(h.wins, 0) + COALESCE(a.wins, 0)       AS wins,
                    COALESCE(h.losses, 0) + COALESCE(a.losses, 0)   AS losses,
                    COALESCE(h.runs_scored, 0) + COALESCE(a.runs_scored, 0)     AS runs_scored,
                    COALESCE(h.runs_allowed, 0) + COALESCE(a.runs_allowed, 0)   AS runs_allowed
                FROM home h
                FULL OUTER JOIN away a ON h.team_name = a.team_name
            )
            SELECT
                team_name,
                wins,
                losses,
                ROUND(wins * 1.0 / GREATEST(wins + losses, 1), 3) AS win_pct,
                runs_scored,
                runs_allowed,
                runs_scored - runs_allowed AS run_diff
            FROM combined
            ORDER BY win_pct DESC, run_diff DESC
        """
        return self._execute(sql, "team_standings", {
            "team_name": pl.Utf8,
            "wins": pl.Int64,
            "losses": pl.Int64,
            "win_pct": pl.Float64,
            "runs_scored": pl.Int64,
            "runs_allowed": pl.Int64,
            "run_diff": pl.Int64,
        })

    def game_pace(self, game_pk: int) -> pl.DataFrame:
        """Pitch-by-pitch pace analysis for a single game.

        Returns per-half-inning stats: pitch count, average time between
        pitches (seconds), and total pitches in the inning.
        """
        glob = self._pitches_glob()

        sql = f"""
            WITH ordered AS (
                SELECT
                    inning,
                    half_inning,
                    timestamp,
                    LAG(timestamp) OVER (
                        PARTITION BY inning, half_inning
                        ORDER BY at_bat_index, pitch_number
                    ) AS prev_ts
                FROM read_parquet('{glob}')
                WHERE game_pk = {game_pk}
            )
            SELECT
                inning,
                half_inning,
                COUNT(*)                                                AS pitch_count,
                ROUND(AVG(
                    CASE WHEN prev_ts IS NOT NULL
                         THEN EXTRACT(EPOCH FROM (timestamp - prev_ts))
                         ELSE NULL
                    END
                ), 1)                                                   AS avg_time_between_pitches,
                COUNT(*)                                                AS total_pitches_in_inning
            FROM ordered
            GROUP BY inning, half_inning
            ORDER BY inning, half_inning
        """
        return self._execute(sql, "game_pace", {
            "inning": pl.Int64,
            "half_inning": pl.Utf8,
            "pitch_count": pl.Int64,
            "avg_time_between_pitches": pl.Float64,
            "total_pitches_in_inning": pl.Int64,
        })

    def daily_pitch_leaders(self, game_date: str) -> pl.DataFrame:
        """Top pitchers for a given date.

        Ranks by strikeouts (from at-bats where ``event_type = 'strikeout'``),
        then by total pitches thrown.
        """
        pitches_glob = self._pitches_glob(game_date)
        ab_glob = self._at_bats_glob(game_date)

        sql = f"""
            WITH pitch_stats AS (
                SELECT
                    pitcher_name,
                    pitcher_id,
                    game_pk,
                    COUNT(*)                AS total_pitches,
                    ROUND(AVG(start_speed), 1)   AS avg_speed,
                    ROUND(MAX(start_speed), 1)   AS max_speed
                FROM read_parquet('{pitches_glob}')
                GROUP BY pitcher_name, pitcher_id, game_pk
            ),
            ab_stats AS (
                SELECT
                    pitcher_id,
                    game_pk,
                    SUM(CASE WHEN event_type = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts
                FROM read_parquet('{ab_glob}')
                GROUP BY pitcher_id, game_pk
            )
            SELECT
                ps.pitcher_name,
                ps.pitcher_id,
                ps.game_pk,
                ps.total_pitches,
                COALESCE(ab.strikeouts, 0) AS strikeouts,
                ps.avg_speed,
                ps.max_speed
            FROM pitch_stats ps
            LEFT JOIN ab_stats ab
                ON ps.pitcher_id = ab.pitcher_id AND ps.game_pk = ab.game_pk
            ORDER BY strikeouts DESC, total_pitches DESC
        """
        return self._execute(sql, "daily_pitch_leaders", {
            "pitcher_name": pl.Utf8,
            "pitcher_id": pl.Int64,
            "game_pk": pl.Int64,
            "total_pitches": pl.Int64,
            "strikeouts": pl.Int64,
            "avg_speed": pl.Float64,
            "max_speed": pl.Float64,
        })

    def season_batting_leaders(
        self,
        season_dates: tuple[str, str],
        min_at_bats: int = 50,
    ) -> pl.DataFrame:
        """Season batting leaders from at-bat Parquet files.

        Filters by the ``(start_date, end_date)`` range (inclusive, ISO format)
        and requires at least *min_at_bats* qualifying at-bats.
        """
        start, end = season_dates
        ab_glob = self._at_bats_glob()

        sql = f"""
            WITH ab AS (
                SELECT
                    *,
                    regexp_extract(filename, '.*/at_bats/([^/]+)/', 1) AS game_date
                FROM read_parquet('{ab_glob}', filename=true)
                WHERE regexp_extract(filename, '.*/at_bats/([^/]+)/', 1) >= '{start}'
                  AND regexp_extract(filename, '.*/at_bats/([^/]+)/', 1) <= '{end}'
            )
            SELECT
                batter_name,
                batter_id,
                COUNT(DISTINCT game_pk)                                     AS games,
                COUNT(*)                                                    AS at_bats,
                SUM(CASE WHEN event_type IN (
                        'single', 'double', 'triple', 'home_run'
                    ) THEN 1 ELSE 0 END)                                    AS hits,
                SUM(CASE WHEN event_type = 'home_run'
                    THEN 1 ELSE 0 END)                                      AS home_runs,
                SUM(rbi)                                                    AS rbi,
                ROUND(
                    SUM(CASE WHEN event_type IN (
                            'single', 'double', 'triple', 'home_run'
                        ) THEN 1 ELSE 0 END)
                    * 1.0 / GREATEST(COUNT(*), 1), 3
                )                                                           AS batting_avg,
                ROUND(
                    (
                        SUM(CASE WHEN event_type = 'single'  THEN 1 ELSE 0 END)
                      + SUM(CASE WHEN event_type = 'double'  THEN 2 ELSE 0 END)
                      + SUM(CASE WHEN event_type = 'triple'  THEN 3 ELSE 0 END)
                      + SUM(CASE WHEN event_type = 'home_run' THEN 4 ELSE 0 END)
                    ) * 1.0 / GREATEST(COUNT(*), 1), 3
                )                                                           AS slug_pct_approx
            FROM ab
            GROUP BY batter_name, batter_id
            HAVING COUNT(*) >= {min_at_bats}
            ORDER BY batting_avg DESC
        """
        return self._execute(sql, "season_batting_leaders", {
            "batter_name": pl.Utf8,
            "batter_id": pl.Int64,
            "games": pl.Int64,
            "at_bats": pl.Int64,
            "hits": pl.Int64,
            "home_runs": pl.Int64,
            "rbi": pl.Int64,
            "batting_avg": pl.Float64,
            "slug_pct_approx": pl.Float64,
        })

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self._conn.close()
        logger.info("duckdb_store_closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(
        self,
        sql: str,
        query_name: str,
        empty_schema: dict[str, pl.DataType],
    ) -> pl.DataFrame:
        """Run *sql* and return a Polars DataFrame.

        If the query fails (e.g. because no Parquet files exist yet), an empty
        DataFrame with *empty_schema* is returned instead.
        """
        try:
            result = self._conn.execute(sql).pl()
            logger.debug("query_ok", query=query_name, rows=len(result))
            return result
        except duckdb.IOException:
            logger.warning("no_parquet_files", query=query_name)
            return pl.DataFrame(schema=empty_schema)
        except duckdb.CatalogException:
            logger.warning("query_catalog_error", query=query_name)
            return pl.DataFrame(schema=empty_schema)
        except Exception:
            logger.exception("query_failed", query=query_name)
            return pl.DataFrame(schema=empty_schema)
