"""Game-level and player-level aggregations using Polars.

All functions are pure transforms on ``pl.LazyFrame`` — no side-effects, no I/O.
Grouping, windowing, and conditional logic use idiomatic Polars expressions.
"""

from __future__ import annotations

import polars as pl
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Column-name constants shared across aggregations
# ---------------------------------------------------------------------------
_HIT_EVENTS = {"single", "double", "triple", "home_run"}
_AB_EXCLUDE_EVENTS = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play"}


# ===================================================================
# 1. Pitcher game-level aggregation
# ===================================================================

def aggregate_pitcher_game(pitches: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate pitch-level data to one row per pitcher per game.

    Expected input columns (from the enriched pitches LazyFrame):
        game_pk, pitcher_id, pitcher_name, pitch_type, start_speed,
        is_strike, is_ball, is_in_play, call_code, inning, half_inning

    Columns produced:
        game_pk, pitcher_id, pitcher_name, total_pitches, strikes, balls,
        whiffs, strike_pct, avg_fastball_speed, max_speed,
        innings_pitched_approx, ff_count, si_count, sl_count, cu_count,
        ch_count, fc_count, fs_count, other_type_count
    """
    is_whiff = pl.col("is_strike") & ~pl.col("is_in_play") & pl.col("call_code").eq("S")
    is_fastball = pl.col("pitch_type").is_in(["FF", "SI"])

    return (
        pitches.group_by("game_pk", "pitcher_id", "pitcher_name")
        .agg(
            pl.len().alias("total_pitches"),
            pl.col("is_strike").sum().cast(pl.Int64).alias("strikes"),
            pl.col("is_ball").sum().cast(pl.Int64).alias("balls"),
            is_whiff.sum().cast(pl.Int64).alias("whiffs"),
            # Strike percentage
            (
                pl.col("is_strike").sum().cast(pl.Float64)
                / pl.len().cast(pl.Float64)
                * 100.0
            ).alias("strike_pct"),
            # Fastball velocities
            pl.col("start_speed")
            .filter(is_fastball)
            .mean()
            .alias("avg_fastball_speed"),
            pl.col("start_speed").max().alias("max_speed"),
            # Approximate innings pitched: distinct (inning, half_inning) pairs
            # Each pair is roughly 0.5 innings contribution by one pitcher, but
            # since a pitcher may not pitch the whole half-inning we just count
            # distinct half-innings as a simple proxy.
            pl.struct("inning", "half_inning")
            .n_unique()
            .cast(pl.Float64)
            .alias("innings_pitched_approx"),
            # Per-type pitch counts
            (pl.col("pitch_type") == "FF").sum().cast(pl.Int64).alias("ff_count"),
            (pl.col("pitch_type") == "SI").sum().cast(pl.Int64).alias("si_count"),
            (pl.col("pitch_type") == "SL").sum().cast(pl.Int64).alias("sl_count"),
            (pl.col("pitch_type") == "CU").sum().cast(pl.Int64).alias("cu_count"),
            (pl.col("pitch_type") == "CH").sum().cast(pl.Int64).alias("ch_count"),
            (pl.col("pitch_type") == "FC").sum().cast(pl.Int64).alias("fc_count"),
            (pl.col("pitch_type") == "FS").sum().cast(pl.Int64).alias("fs_count"),
            (
                ~pl.col("pitch_type").is_in(["FF", "SI", "SL", "CU", "CH", "FC", "FS"])
                & pl.col("pitch_type").is_not_null()
            )
            .sum()
            .cast(pl.Int64)
            .alias("other_type_count"),
        )
    )


# ===================================================================
# 2. Batter game-level aggregation
# ===================================================================

def aggregate_batter_game(
    at_bats: pl.LazyFrame,
    pitches: pl.LazyFrame,
) -> pl.LazyFrame:
    """Aggregate at-bat outcomes to one row per batter per game.

    ``at_bats`` expected columns (from AtBatResult):
        game_pk, batter_id, batter_name, event_type, rbi, half_inning

    ``pitches`` is unused for the primary stats but kept in the signature for
    future enrichment (e.g. pitches-seen per AB).

    Columns produced:
        game_pk, batter_id, batter_name, plate_appearances, at_bats_count,
        hits, doubles, triples, home_runs, rbi, strikeouts, walks,
        batting_avg
    """
    et = pl.col("event_type")

    is_hit = et.is_in(list(_HIT_EVENTS))
    is_ab_excluded = et.is_in(list(_AB_EXCLUDE_EVENTS))

    return (
        at_bats.group_by("game_pk", "batter_id", "batter_name")
        .agg(
            pl.len().alias("plate_appearances"),
            (~is_ab_excluded).sum().cast(pl.Int64).alias("at_bats_count"),
            is_hit.sum().cast(pl.Int64).alias("hits"),
            (et == "double").sum().cast(pl.Int64).alias("doubles"),
            (et == "triple").sum().cast(pl.Int64).alias("triples"),
            (et == "home_run").sum().cast(pl.Int64).alias("home_runs"),
            pl.col("rbi").sum().alias("rbi"),
            (et == "strikeout").sum().cast(pl.Int64).alias("strikeouts"),
            (et == "walk").sum().cast(pl.Int64).alias("walks"),
        )
        .with_columns(
            pl.when(pl.col("at_bats_count") > 0)
            .then(
                pl.col("hits").cast(pl.Float64) / pl.col("at_bats_count").cast(pl.Float64)
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("batting_avg"),
        )
    )


# ===================================================================
# 3. Team game-level aggregation
# ===================================================================

def aggregate_team_game(at_bats: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate at-bat outcomes per team (batting side) per game.

    ``half_inning`` encodes which team is batting:
        - "top"    -> away team batting
        - "bottom" -> home team batting

    Group by game_pk, half_inning to separate home/away.

    Columns produced:
        game_pk, half_inning, total_hits, total_runs (sum of rbi as proxy),
        total_rbi, total_strikeouts, total_walks, total_home_runs
    """
    et = pl.col("event_type")

    return (
        at_bats.group_by("game_pk", "half_inning")
        .agg(
            et.is_in(list(_HIT_EVENTS)).sum().cast(pl.Int64).alias("total_hits"),
            pl.col("rbi").sum().alias("total_rbi"),
            # total_runs: use away_score/home_score max per half or rbi as proxy
            pl.col("is_scoring_play").sum().cast(pl.Int64).alias("total_scoring_plays"),
            (et == "strikeout").sum().cast(pl.Int64).alias("total_strikeouts"),
            (et == "walk").sum().cast(pl.Int64).alias("total_walks"),
            (et == "home_run").sum().cast(pl.Int64).alias("total_home_runs"),
        )
        .with_columns(
            # Use total_rbi as the best available run proxy at this level
            pl.col("total_rbi").alias("total_runs"),
        )
    )


# ===================================================================
# 4. Rolling player stats
# ===================================================================

def rolling_player_stats(
    daily_stats: pl.LazyFrame,
    window: int = 15,
) -> pl.LazyFrame:
    """Compute rolling window statistics over a player's daily game logs.

    Expected input columns (one row per player per game-date):
        player_id, stat_date, hits, at_bats_count, strikeouts, walks,
        plate_appearances

    Columns added:
        rolling_avg        — hits / at_bats over the rolling window
        rolling_k_rate     — strikeouts / plate_appearances
        rolling_bb_rate    — walks / plate_appearances
        rolling_ops_approx — simplified OPS approximation:
                             (hits + walks) / (at_bats + walks)  (OBP-like)
                             + hits / at_bats                     (SLG-like,
                             ignoring extra-base weight for simplicity)
    """
    partition = "player_id"
    order = "stat_date"

    rolling_hits = (
        pl.col("hits")
        .rolling_sum(window_size=window, min_samples=1)
        .over(partition, order_by=order)
    )
    rolling_abs = (
        pl.col("at_bats_count")
        .rolling_sum(window_size=window, min_samples=1)
        .over(partition, order_by=order)
    )
    rolling_k = (
        pl.col("strikeouts")
        .rolling_sum(window_size=window, min_samples=1)
        .over(partition, order_by=order)
    )
    rolling_bb = (
        pl.col("walks")
        .rolling_sum(window_size=window, min_samples=1)
        .over(partition, order_by=order)
    )
    rolling_pa = (
        pl.col("plate_appearances")
        .rolling_sum(window_size=window, min_samples=1)
        .over(partition, order_by=order)
    )

    safe_abs = pl.when(rolling_abs > 0).then(rolling_abs.cast(pl.Float64)).otherwise(pl.lit(None, dtype=pl.Float64))
    safe_pa = pl.when(rolling_pa > 0).then(rolling_pa.cast(pl.Float64)).otherwise(pl.lit(None, dtype=pl.Float64))
    safe_abs_bb = pl.when((rolling_abs + rolling_bb) > 0).then((rolling_abs + rolling_bb).cast(pl.Float64)).otherwise(pl.lit(None, dtype=pl.Float64))

    return daily_stats.with_columns(
        (rolling_hits.cast(pl.Float64) / safe_abs).alias("rolling_avg"),
        (rolling_k.cast(pl.Float64) / safe_pa).alias("rolling_k_rate"),
        (rolling_bb.cast(pl.Float64) / safe_pa).alias("rolling_bb_rate"),
        # Simplified OPS: OBP-approx + SLG-approx
        (
            (rolling_hits + rolling_bb).cast(pl.Float64) / safe_abs_bb
            + rolling_hits.cast(pl.Float64) / safe_abs
        ).alias("rolling_ops_approx"),
    )


# ===================================================================
# 5. Game summary — one row per game
# ===================================================================

def game_summary(
    games: pl.LazyFrame,
    pitches: pl.LazyFrame,
    at_bats: pl.LazyFrame,
) -> pl.LazyFrame:
    """Produce a single summary row per game.

    ``games`` expected columns:
        game_pk, game_date, away_team_name, home_team_name, away_score,
        home_score

    ``pitches`` and ``at_bats`` supply pitch/at-bat level aggregates.

    Columns produced:
        game_pk, game_date, away_team, home_team, away_score, home_score,
        winner, total_pitches, total_at_bats, total_hits, total_home_runs,
        total_strikeouts, game_duration_pitches
    """
    et = pl.col("event_type")

    pitch_agg = (
        pitches.group_by("game_pk")
        .agg(
            pl.len().alias("total_pitches"),
        )
    )

    ab_agg = (
        at_bats.group_by("game_pk")
        .agg(
            pl.len().alias("total_at_bats"),
            et.is_in(list(_HIT_EVENTS)).sum().cast(pl.Int64).alias("total_hits"),
            (et == "home_run").sum().cast(pl.Int64).alias("total_home_runs"),
            (et == "strikeout").sum().cast(pl.Int64).alias("total_strikeouts"),
        )
    )

    summary = (
        games.select(
            "game_pk",
            "game_date",
            pl.col("away_team_name").alias("away_team"),
            pl.col("home_team_name").alias("home_team"),
            "away_score",
            "home_score",
        )
        .join(pitch_agg, on="game_pk", how="left")
        .join(ab_agg, on="game_pk", how="left")
        .with_columns(
            pl.when(pl.col("home_score") > pl.col("away_score"))
            .then(pl.col("home_team"))
            .when(pl.col("away_score") > pl.col("home_score"))
            .then(pl.col("away_team"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .alias("winner"),
            # game_duration_pitches is the same as total_pitches — a simple
            # proxy for game length when wall-clock time is unavailable.
            pl.col("total_pitches").alias("game_duration_pitches"),
        )
    )

    return summary
