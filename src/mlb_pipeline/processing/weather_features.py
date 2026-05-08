"""Weather feature engineering for the win-probability model.

Joins game_weather + venues + games into a per-game feature row, then derives
ballpark-relative wind components, temperature bins, and an indoor neutralization
flag so the model can learn to ignore weather inside domes / under closed roofs.
"""

from __future__ import annotations

import math

import polars as pl
import structlog
from sqlalchemy.engine import Engine

logger = structlog.get_logger()


_INDOOR_CONDITIONS = {"Roof Closed", "Dome", "Indoors"}


def load_weather_frame(engine: Engine) -> pl.DataFrame:
    """Read game_weather joined with venues and games into a Polars DataFrame.

    One row per game with all weather + venue location columns. Indoor / closed-roof
    games are included with raw weather kept; downstream features should respect
    ``is_indoor_or_closed`` to neutralize values when desired.
    """
    sql = """
    SELECT
        gw.game_pk,
        g.game_date,
        g.first_pitch_time_utc,
        v.venue_id,
        v.name              AS venue_name,
        v.latitude,
        v.longitude,
        v.elevation_ft,
        v.azimuth_angle,
        v.roof_type,
        gw.is_indoor,
        gw.roof_closed,
        gw.temperature_f,
        gw.feels_like_f,
        gw.relative_humidity_pct,
        gw.precipitation_in,
        gw.wind_speed_mph,
        gw.wind_gust_mph,
        gw.wind_direction_deg,
        gw.cloud_cover_pct,
        gw.pressure_hpa,
        gw.weather_code,
        gw.mlb_condition
    FROM game_weather gw
    JOIN games   g ON g.game_pk   = gw.game_pk
    LEFT JOIN venues v ON v.venue_id = gw.venue_id
    """
    with engine.connect() as conn:
        return pl.read_database(query=sql, connection=conn)


def add_indoor_flag(df: pl.DataFrame) -> pl.DataFrame:
    """Single boolean: weather has no on-field effect (dome or roof closed)."""
    return df.with_columns(
        (pl.col("is_indoor") | pl.col("roof_closed").fill_null(False))
        .alias("is_indoor_or_closed")
    )


def add_wind_components(df: pl.DataFrame) -> pl.DataFrame:
    """Project wind onto the home-plate→centerfield axis using venue azimuth.

    azimuth_angle is the bearing (deg clockwise from N) from home plate to
    second base (≈ CF). wind_direction_deg is the direction wind blows FROM
    (meteorological convention). Wind direction TOWARD CF = azimuth_angle.

    wind_out_to_cf_mph: positive = blowing out toward CF (helps fly balls),
                       negative = blowing in toward home (suppresses fly balls)
    wind_cross_mph:    positive = blowing toward right field, negative = LF
    """
    delta_rad = (
        (pl.col("wind_direction_deg") - pl.col("azimuth_angle"))
        * (math.pi / 180.0)
    )

    return df.with_columns(
        (-pl.col("wind_speed_mph") * delta_rad.cos()).alias("wind_out_to_cf_mph"),
        (pl.col("wind_speed_mph") * delta_rad.sin()).alias("wind_cross_mph"),
    )


def add_temperature_features(df: pl.DataFrame) -> pl.DataFrame:
    """Categorical temperature bins + a precipitation indicator."""
    temp = pl.col("temperature_f")
    return df.with_columns(
        pl.when(temp < 50)
        .then(pl.lit("cold"))
        .when(temp < 65)
        .then(pl.lit("cool"))
        .when(temp < 80)
        .then(pl.lit("mild"))
        .when(temp < 90)
        .then(pl.lit("warm"))
        .otherwise(pl.lit("hot"))
        .alias("temp_band"),
        (pl.col("precipitation_in").fill_null(0.0) > 0.05).alias("is_wet"),
    )


def neutralize_indoor(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``effective_*`` columns that zero out weather inside domes/closed roofs.

    The model gets two parallel views:
      - raw columns (temperature_f, wind_speed_mph, ...) for outdoor analysis
      - effective_* columns + is_indoor_or_closed flag for the model input
    """
    indoor = pl.col("is_indoor_or_closed")
    neutral_temp = pl.lit(72.0)  # comfortable indoor-ish baseline

    return df.with_columns(
        pl.when(indoor)
        .then(neutral_temp)
        .otherwise(pl.col("temperature_f"))
        .alias("effective_temperature_f"),
        pl.when(indoor)
        .then(pl.lit(0.0))
        .otherwise(pl.col("wind_speed_mph"))
        .alias("effective_wind_speed_mph"),
        pl.when(indoor)
        .then(pl.lit(0.0))
        .otherwise(pl.col("wind_out_to_cf_mph"))
        .alias("effective_wind_out_to_cf_mph"),
        pl.when(indoor)
        .then(pl.lit(0.0))
        .otherwise(pl.col("wind_cross_mph"))
        .alias("effective_wind_cross_mph"),
        pl.when(indoor)
        .then(pl.lit(0.0))
        .otherwise(pl.col("precipitation_in").fill_null(0.0))
        .alias("effective_precipitation_in"),
    )


def build_weather_features(engine: Engine) -> pl.DataFrame:
    """Full pipeline: load -> indoor flag -> wind components -> temp bins -> neutralize.

    Returns a per-game DataFrame ready to join onto the win-probability training
    matrix on ``game_pk``.
    """
    df = load_weather_frame(engine)
    df = add_indoor_flag(df)
    df = add_wind_components(df)
    df = add_temperature_features(df)
    df = neutralize_indoor(df)
    return df


WEATHER_FEATURE_COLUMNS: list[str] = [
    "game_pk",
    "is_indoor_or_closed",
    "effective_temperature_f",
    "effective_wind_speed_mph",
    "effective_wind_out_to_cf_mph",
    "effective_wind_cross_mph",
    "effective_precipitation_in",
    "elevation_ft",
    "relative_humidity_pct",
]
"""Minimal column set the model should consume for weather context."""
