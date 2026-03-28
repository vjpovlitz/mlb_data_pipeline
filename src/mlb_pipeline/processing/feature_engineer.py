"""Feature engineering for win probability models.

Transforms raw at-bat events into numeric feature vectors suitable for ML.
Uses Polars for fast, lazy evaluation.

Feature vector (per at-bat):
  [0]  inning_norm          — inning / 9 (clipped to 1.0 in extras)
  [1]  is_bottom            — 1 if bottom half, 0 if top
  [2]  score_diff           — home_score - away_score (home team perspective)
  [3]  score_diff_clip      — clipped to [-5, 5] / 5
  [4]  outs_norm            — outs / 2
  [5]  runner_on_first      — bool
  [6]  runner_on_second     — bool
  [7]  runner_on_third      — bool
  [8]  base_state           — 0-7 encoding of 3-bit runner state
  [9]  leverage_index       — proxy: outs * base_state pressure score
  [10] home_batting          — 1 if home team is batting
  [11] innings_remaining     — (9 - inning) / 9 (capped 0)
  [12] is_late_game          — 1 if inning >= 7
  [13] is_extras             — 1 if inning > 9
  [14] score_tied            — 1 if score_diff == 0
  [15] home_leading          — 1 if score_diff > 0
"""

from __future__ import annotations

import numpy as np
import polars as pl


# ------------------------------------------------------------------
# Scalar feature extraction (used during real-time inference)
# ------------------------------------------------------------------

def extract_features_from_at_bat(
    inning: int,
    half_inning: str,
    outs: int,
    away_score: int,
    home_score: int,
    runner_on_first: bool,
    runner_on_second: bool,
    runner_on_third: bool,
) -> np.ndarray:
    """Convert a single game state into a float32 feature vector."""
    is_bottom = 1.0 if half_inning in ("bottom", "BOTTOM") else 0.0
    score_diff = float(home_score - away_score)
    runner_state = int(runner_on_first) | (int(runner_on_second) << 1) | (int(runner_on_third) << 2)
    innings_remaining = max(0.0, (9.0 - inning) / 9.0)

    features = np.array(
        [
            min(inning / 9.0, 1.5),            # 0
            is_bottom,                          # 1
            score_diff,                         # 2
            np.clip(score_diff / 5.0, -1.0, 1.0),  # 3
            outs / 2.0,                         # 4
            float(runner_on_first),             # 5
            float(runner_on_second),            # 6
            float(runner_on_third),             # 7
            runner_state / 7.0,                 # 8
            (outs * runner_state) / 14.0,       # 9  leverage proxy
            is_bottom,                          # 10 home batting when bottom
            innings_remaining,                  # 11
            1.0 if inning >= 7 else 0.0,        # 12
            1.0 if inning > 9 else 0.0,         # 13
            1.0 if score_diff == 0 else 0.0,    # 14
            1.0 if score_diff > 0 else 0.0,     # 15
        ],
        dtype=np.float32,
    )
    return features


FEATURE_DIM = 16


# ------------------------------------------------------------------
# Batch feature engineering via Polars (for training dataset creation)
# ------------------------------------------------------------------

def build_training_dataset(at_bats_df: pl.DataFrame, games_df: pl.DataFrame) -> pl.DataFrame:
    """Join at-bat rows with game outcomes and engineer all features.

    Args:
        at_bats_df: Polars DataFrame with columns matching AtBatResult fields.
        games_df: Polars DataFrame with columns game_pk and home_team_won (bool).

    Returns:
        DataFrame with feature columns + 'label' (1=home won, 0=home lost).
    """
    # Join game outcome
    df = at_bats_df.join(
        games_df.select(["game_pk", "home_team_won"]),
        on="game_pk",
        how="inner",
    ).filter(pl.col("home_team_won").is_not_null())

    df = df.with_columns([
        # inning features
        (pl.col("inning") / 9.0).clip(0.0, 1.5).alias("inning_norm"),
        pl.when(pl.col("half_inning") == "bottom").then(1.0).otherwise(0.0).alias("is_bottom"),

        # score
        (pl.col("home_score") - pl.col("away_score")).alias("score_diff"),
        ((pl.col("home_score") - pl.col("away_score")) / 5.0).clip(-1.0, 1.0).alias("score_diff_clip"),

        # count
        (pl.col("outs_after") / 2.0).alias("outs_norm"),

        # runners
        pl.col("runner_on_first").cast(pl.Float32),
        pl.col("runner_on_second").cast(pl.Float32),
        pl.col("runner_on_third").cast(pl.Float32),

        # base state 0-7
        (
            pl.col("runner_on_first").cast(pl.Int32)
            | (pl.col("runner_on_second").cast(pl.Int32) * 2)
            | (pl.col("runner_on_third").cast(pl.Int32) * 4)
        ).cast(pl.Float32).alias("base_state_norm") / 7.0,

        # leverage proxy
        (pl.col("outs_after") * (
            pl.col("runner_on_first").cast(pl.Int32)
            | (pl.col("runner_on_second").cast(pl.Int32) * 2)
            | (pl.col("runner_on_third").cast(pl.Int32) * 4)
        ) / 14.0).cast(pl.Float32).alias("leverage"),

        # game situation
        ((9.0 - pl.col("inning")).clip(lower_bound=0.0) / 9.0).alias("innings_remaining"),
        pl.when(pl.col("inning") >= 7).then(1.0).otherwise(0.0).alias("is_late_game"),
        pl.when(pl.col("inning") > 9).then(1.0).otherwise(0.0).alias("is_extras"),
        pl.when(pl.col("home_score") == pl.col("away_score")).then(1.0).otherwise(0.0).alias("score_tied"),
        pl.when(pl.col("home_score") > pl.col("away_score")).then(1.0).otherwise(0.0).alias("home_leading"),

        # label
        pl.col("home_team_won").cast(pl.Float32).alias("label"),
    ])

    feature_cols = [
        "inning_norm", "is_bottom", "score_diff", "score_diff_clip",
        "outs_norm", "runner_on_first", "runner_on_second", "runner_on_third",
        "base_state_norm", "leverage", "innings_remaining",
        "is_late_game", "is_extras", "score_tied", "home_leading",
        "home_score", "away_score",  # raw scores as extras for LSTM context
        "label",
    ]

    return df.select(["game_pk", "at_bat_index", "inning", "timestamp"] + feature_cols)


TRAINING_FEATURE_COLS = [
    "inning_norm", "is_bottom", "score_diff", "score_diff_clip",
    "outs_norm", "runner_on_first", "runner_on_second", "runner_on_third",
    "base_state_norm", "leverage", "innings_remaining",
    "is_late_game", "is_extras", "score_tied", "home_leading",
    "home_score", "away_score",
]

TRAINING_FEATURE_DIM = len(TRAINING_FEATURE_COLS)  # 17
