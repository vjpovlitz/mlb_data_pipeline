"""Polars transforms for pitch-level and game-level data enrichment.

All functions accept and return ``pl.LazyFrame`` so they compose cleanly in a
pipeline.  No side-effects, no I/O — pure column-level transforms only.
"""

from __future__ import annotations

import polars as pl
import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Zone geometry constants (feet, relative to center of plate)
# ---------------------------------------------------------------------------
_ZONE_X_INNER = 0.28  # inner third boundary (plate_x absolute)
_ZONE_X_OUTER = 0.83  # outer edge of strike zone
_ZONE_Z_LOW = 1.5
_ZONE_Z_HIGH = 3.5
_ZONE_Z_MID_LOW = 2.17  # lower third boundary
_ZONE_Z_MID_HIGH = 2.83  # upper third boundary
_CHASE_BUFFER = 0.5  # feet outside zone that counts as "chase"


# ===================================================================
# 1. Pitch-sequence features (within an at-bat)
# ===================================================================

def add_pitch_sequence_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add within-at-bat pitch sequence features.

    Columns added:
        prev_pitch_type  — pitch type of the previous pitch (null for first)
        prev_speed       — start_speed of the previous pitch
        speed_diff       — difference from previous pitch speed
        pitch_number_in_ab — 1-based running pitch count in the at-bat
    """
    partition = ["game_pk", "at_bat_index"]

    return lf.with_columns(
        pl.col("pitch_type")
        .shift(1)
        .over(partition, order_by="pitch_number")
        .alias("prev_pitch_type"),
        pl.col("start_speed")
        .shift(1)
        .over(partition, order_by="pitch_number")
        .alias("prev_speed"),
    ).with_columns(
        (pl.col("start_speed") - pl.col("prev_speed")).alias("speed_diff"),
        pl.col("pitch_number")
        .rank("ordinal")
        .over(partition, order_by="pitch_number")
        .cast(pl.Int32)
        .alias("pitch_number_in_ab"),
    )


# ===================================================================
# 2. Zone features — pitch location classification
# ===================================================================

def add_zone_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Classify pitch location into zone-based features.

    Columns added:
        is_in_zone          — boolean, True when inside the rule-book zone
        zone_region         — "heart", "edge", "chase", or "waste"
        horizontal_location — "inside", "middle", or "outside"
        vertical_location   — "high", "middle", or "low"
    """
    px = pl.col("plate_x")
    pz = pl.col("plate_z")
    abs_px = px.abs()

    in_zone = (
        (abs_px <= _ZONE_X_OUTER)
        & (pz >= _ZONE_Z_LOW)
        & (pz <= _ZONE_Z_HIGH)
    )

    # Heart: inner third on both axes
    is_heart = (
        (abs_px <= _ZONE_X_INNER)
        & (pz >= _ZONE_Z_MID_LOW)
        & (pz <= _ZONE_Z_MID_HIGH)
    )

    # Chase: outside the zone but within CHASE_BUFFER of the edge
    chase_x_limit = _ZONE_X_OUTER + _CHASE_BUFFER
    chase_z_low = _ZONE_Z_LOW - _CHASE_BUFFER
    chase_z_high = _ZONE_Z_HIGH + _CHASE_BUFFER

    is_chase = (
        ~in_zone
        & (abs_px <= chase_x_limit)
        & (pz >= chase_z_low)
        & (pz <= chase_z_high)
    )

    zone_region = (
        pl.when(px.is_null() | pz.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(is_heart)
        .then(pl.lit("heart"))
        .when(in_zone)
        .then(pl.lit("edge"))
        .when(is_chase)
        .then(pl.lit("chase"))
        .otherwise(pl.lit("waste"))
    )

    horizontal = (
        pl.when(px.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(abs_px <= _ZONE_X_INNER)
        .then(pl.lit("middle"))
        .when(px < -_ZONE_X_INNER)
        .then(pl.lit("inside"))
        .otherwise(pl.lit("outside"))
    )

    vertical = (
        pl.when(pz.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pz >= _ZONE_Z_MID_HIGH)
        .then(pl.lit("high"))
        .when(pz <= _ZONE_Z_MID_LOW)
        .then(pl.lit("low"))
        .otherwise(pl.lit("middle"))
    )

    return lf.with_columns(
        in_zone.alias("is_in_zone"),
        zone_region.alias("zone_region"),
        horizontal.alias("horizontal_location"),
        vertical.alias("vertical_location"),
    )


# ===================================================================
# 3. Game-context features
# ===================================================================

def add_game_context(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add game-situation features derived from score, runners, and inning.

    Columns added:
        score_diff          — home_score - away_score
        is_close_game       — abs(score_diff) <= 2
        base_state          — bitmask (1B=1, 2B=2, 3B=4)
        runners_on_base     — 0-3 count of occupied bases
        is_scoring_position — runner on 2nd or 3rd
        leverage_bucket     — "high", "medium", or "low"
    """
    score_diff = pl.col("home_score") - pl.col("away_score")
    abs_diff = score_diff.abs()

    r1 = pl.col("runner_on_first").cast(pl.Int32)
    r2 = pl.col("runner_on_second").cast(pl.Int32)
    r3 = pl.col("runner_on_third").cast(pl.Int32)

    base_state = r1 + r2 * 2 + r3 * 4
    runners_on_base = r1 + r2 + r3
    is_scoring_pos = pl.col("runner_on_second") | pl.col("runner_on_third")

    is_close = abs_diff <= 2
    is_late = pl.col("inning") >= 7

    leverage = (
        pl.when(is_close & is_late & is_scoring_pos)
        .then(pl.lit("high"))
        .when(is_close | (runners_on_base > 0))
        .then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
    )

    return lf.with_columns(
        score_diff.alias("score_diff"),
        is_close.alias("is_close_game"),
        base_state.alias("base_state"),
        runners_on_base.alias("runners_on_base"),
        is_scoring_pos.alias("is_scoring_position"),
        leverage.alias("leverage_bucket"),
    )


# ===================================================================
# 4. Count features
# ===================================================================

def add_count_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add count-based features.

    Columns added:
        is_first_pitch  — pitch_number == 1
        is_full_count   — 3-2 count
        is_hitter_count — balls > strikes
        is_pitcher_count — strikes > balls
        count_label     — e.g. "2-1"
    """
    balls = pl.col("balls")
    strikes = pl.col("strikes")

    return lf.with_columns(
        (pl.col("pitch_number") == 1).alias("is_first_pitch"),
        ((balls == 3) & (strikes == 2)).alias("is_full_count"),
        (balls > strikes).alias("is_hitter_count"),
        (strikes > balls).alias("is_pitcher_count"),
        (balls.cast(pl.Utf8) + pl.lit("-") + strikes.cast(pl.Utf8)).alias(
            "count_label"
        ),
    )


# ===================================================================
# 5. Convenience: apply all enrichments
# ===================================================================

def enrich_pitches(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Apply every pitch-level transform in the recommended order.

    Equivalent to piping through each ``add_*`` function sequentially.
    """
    return (
        lf.pipe(add_pitch_sequence_features)
        .pipe(add_zone_features)
        .pipe(add_game_context)
        .pipe(add_count_features)
    )
