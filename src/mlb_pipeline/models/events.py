"""Pydantic event models — the canonical data contract for the pipeline.

Every component produces or consumes these shapes. The ingestion layer creates them
from raw MLB API JSON, Redis transports their serialization, consumers deserialize
them, and storage maps them to database rows.
"""

from datetime import datetime

from pydantic import BaseModel

from .enums import GameState, HalfInning


class PitchEvent(BaseModel):
    """Single pitch within an at-bat. Core unit of the pipeline."""

    game_pk: int
    event_id: str  # f"{game_pk}_{at_bat_index}_{pitch_number}"
    timestamp: datetime
    inning: int
    half_inning: HalfInning
    at_bat_index: int
    pitch_number: int

    # Participants
    pitcher_id: int
    pitcher_name: str
    batter_id: int
    batter_name: str

    # Pitch characteristics
    pitch_type: str | None = None
    start_speed: float | None = None
    end_speed: float | None = None
    spin_rate: float | None = None
    spin_direction: float | None = None
    break_angle: float | None = None
    break_length: float | None = None
    pfx_x: float | None = None
    pfx_z: float | None = None
    plate_x: float | None = None
    plate_z: float | None = None
    zone: int | None = None

    # Pitch result
    call_code: str  # "B", "S", "X", "F", etc.
    call_description: str
    is_in_play: bool
    is_strike: bool
    is_ball: bool

    # Hit data (populated only when is_in_play=True)
    launch_speed: float | None = None
    launch_angle: float | None = None
    total_distance: float | None = None
    trajectory: str | None = None

    # Count state AFTER this pitch
    balls: int
    strikes: int
    outs: int

    # Base runners AFTER this pitch
    runner_on_first: bool = False
    runner_on_second: bool = False
    runner_on_third: bool = False

    # Score at time of pitch
    away_score: int = 0
    home_score: int = 0


class AtBatResult(BaseModel):
    """Completed at-bat outcome."""

    game_pk: int
    at_bat_index: int
    timestamp: datetime
    inning: int
    half_inning: HalfInning

    pitcher_id: int
    pitcher_name: str
    batter_id: int
    batter_name: str

    event: str  # "Single", "Strikeout", "Home Run", etc.
    event_type: str  # "single", "strikeout", "home_run"
    description: str
    rbi: int
    away_score: int
    home_score: int
    is_scoring_play: bool

    outs_after: int
    pitch_count: int  # number of pitches in this at-bat

    runner_on_first: bool = False
    runner_on_second: bool = False
    runner_on_third: bool = False


class GameStateEvent(BaseModel):
    """Game-level state change (start, end, delay, etc.)."""

    game_pk: int
    timestamp: datetime
    previous_state: GameState
    new_state: GameState
    away_team: str | None = None
    home_team: str | None = None
    away_score: int = 0
    home_score: int = 0
    detail: str | None = None
