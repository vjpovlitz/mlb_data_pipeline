"""SQLAlchemy ORM models for SQL Server."""

from datetime import UTC, date, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Game(Base):
    __tablename__ = "games"

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    game_date: Mapped[date] = mapped_column()
    game_type: Mapped[str] = mapped_column(String(2))  # "R", "P", "W", "S", etc.
    season: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))  # Preview, Live, Final

    away_team_id: Mapped[int] = mapped_column(Integer)
    away_team_name: Mapped[str] = mapped_column(String(100))
    home_team_id: Mapped[int] = mapped_column(Integer)
    home_team_name: Mapped[str] = mapped_column(String(100))
    venue_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    away_score: Mapped[int] = mapped_column(Integer, default=0)
    home_score: Mapped[int] = mapped_column(Integer, default=0)
    home_team_won: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None),
        onupdate=lambda: datetime.now(UTC).replace(tzinfo=None),
    )


class Pitch(Base):
    __tablename__ = "pitches"
    __table_args__ = (
        Index("ix_pitches_game_pk", "game_pk"),
        Index("ix_pitches_pitcher_ts", "pitcher_id", "timestamp"),
        Index("ix_pitches_batter_ts", "batter_id", "timestamp"),
    )

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    at_bat_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    pitch_number: Mapped[int] = mapped_column(Integer, primary_key=True)

    event_id: Mapped[str] = mapped_column(String(50), unique=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    inning: Mapped[int] = mapped_column(Integer)
    half_inning: Mapped[str] = mapped_column(String(6))

    pitcher_id: Mapped[int] = mapped_column(Integer)
    pitcher_name: Mapped[str] = mapped_column(String(100))
    batter_id: Mapped[int] = mapped_column(Integer)
    batter_name: Mapped[str] = mapped_column(String(100))

    pitch_type: Mapped[str | None] = mapped_column(String(5), nullable=True)
    start_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    spin_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    spin_direction: Mapped[float | None] = mapped_column(Float, nullable=True)
    break_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    break_length: Mapped[float | None] = mapped_column(Float, nullable=True)
    pfx_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    pfx_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    plate_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone: Mapped[int | None] = mapped_column(Integer, nullable=True)

    call_code: Mapped[str] = mapped_column(String(5))
    call_description: Mapped[str] = mapped_column(String(100))
    is_in_play: Mapped[bool] = mapped_column(Boolean)
    is_strike: Mapped[bool] = mapped_column(Boolean)
    is_ball: Mapped[bool] = mapped_column(Boolean)

    launch_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    launch_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    trajectory: Mapped[str | None] = mapped_column(String(50), nullable=True)

    balls: Mapped[int] = mapped_column(Integer)
    strikes: Mapped[int] = mapped_column(Integer)
    outs: Mapped[int] = mapped_column(Integer)

    runner_on_first: Mapped[bool] = mapped_column(Boolean, default=False)
    runner_on_second: Mapped[bool] = mapped_column(Boolean, default=False)
    runner_on_third: Mapped[bool] = mapped_column(Boolean, default=False)

    away_score: Mapped[int] = mapped_column(Integer, default=0)
    home_score: Mapped[int] = mapped_column(Integer, default=0)


class AtBat(Base):
    __tablename__ = "at_bats"
    __table_args__ = (
        Index("ix_at_bats_game_pk", "game_pk"),
        Index("ix_at_bats_pitcher_id", "pitcher_id"),
        Index("ix_at_bats_batter_id", "batter_id"),
    )

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    at_bat_index: Mapped[int] = mapped_column(Integer, primary_key=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime)
    inning: Mapped[int] = mapped_column(Integer)
    half_inning: Mapped[str] = mapped_column(String(6))

    pitcher_id: Mapped[int] = mapped_column(Integer)
    pitcher_name: Mapped[str] = mapped_column(String(100))
    batter_id: Mapped[int] = mapped_column(Integer)
    batter_name: Mapped[str] = mapped_column(String(100))

    event: Mapped[str] = mapped_column(String(100))
    event_type: Mapped[str] = mapped_column(String(50))
    description: Mapped[str] = mapped_column(Text)
    rbi: Mapped[int] = mapped_column(Integer)
    away_score: Mapped[int] = mapped_column(Integer)
    home_score: Mapped[int] = mapped_column(Integer)
    is_scoring_play: Mapped[bool] = mapped_column(Boolean)

    outs_after: Mapped[int] = mapped_column(Integer)
    pitch_count: Mapped[int] = mapped_column(Integer)

    runner_on_first: Mapped[bool] = mapped_column(Boolean, default=False)
    runner_on_second: Mapped[bool] = mapped_column(Boolean, default=False)
    runner_on_third: Mapped[bool] = mapped_column(Boolean, default=False)


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name: Mapped[str] = mapped_column(String(100))
    abbreviation: Mapped[str] = mapped_column(String(10))
    league: Mapped[str] = mapped_column(String(50))
    division: Mapped[str] = mapped_column(String(50))
    venue_name: Mapped[str | None] = mapped_column(String(200), nullable=True)


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    full_name: Mapped[str] = mapped_column(String(100))
    primary_position: Mapped[str] = mapped_column(String(5))
    bat_side: Mapped[str | None] = mapped_column(String(1), nullable=True)
    pitch_hand: Mapped[str | None] = mapped_column(String(1), nullable=True)
    current_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PlayerStatsDaily(Base):
    __tablename__ = "player_stats_daily"
    __table_args__ = (
        Index("ix_player_stats_daily_player_date", "player_id", "stat_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer)
    stat_date: Mapped[date] = mapped_column()

    games: Mapped[int] = mapped_column(Integer, default=0)
    plate_appearances: Mapped[int] = mapped_column(Integer, default=0)
    at_bats: Mapped[int] = mapped_column(Integer, default=0)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    doubles: Mapped[int] = mapped_column(Integer, default=0)
    triples: Mapped[int] = mapped_column(Integer, default=0)
    home_runs: Mapped[int] = mapped_column(Integer, default=0)
    rbi: Mapped[int] = mapped_column(Integer, default=0)
    walks: Mapped[int] = mapped_column(Integer, default=0)
    strikeouts: Mapped[int] = mapped_column(Integer, default=0)

    # Pitching stats
    innings_pitched: Mapped[float] = mapped_column(Float, default=0.0)
    pitches_thrown: Mapped[int] = mapped_column(Integer, default=0)
    earned_runs: Mapped[int] = mapped_column(Integer, default=0)
    strikeouts_pitching: Mapped[int] = mapped_column(Integer, default=0)
    walks_pitching: Mapped[int] = mapped_column(Integer, default=0)


class WinProbabilityLog(Base):
    __tablename__ = "win_probability_log"
    __table_args__ = (
        Index("ix_winprob_game_pk", "game_pk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    inning: Mapped[int] = mapped_column(Integer)
    half_inning: Mapped[str] = mapped_column(String(6))
    outs: Mapped[int] = mapped_column(Integer)
    away_score: Mapped[int] = mapped_column(Integer)
    home_score: Mapped[int] = mapped_column(Integer)
    runner_on_first: Mapped[bool] = mapped_column(Boolean)
    runner_on_second: Mapped[bool] = mapped_column(Boolean)
    runner_on_third: Mapped[bool] = mapped_column(Boolean)
    home_win_probability: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
