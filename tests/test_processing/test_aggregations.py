"""Tests for Polars aggregation functions."""

import polars as pl
import pytest

from mlb_pipeline.processing.aggregations import (
    aggregate_batter_game,
    aggregate_pitcher_game,
    aggregate_team_game,
    game_summary,
    rolling_player_stats,
)


@pytest.fixture
def pitches_lf() -> pl.LazyFrame:
    """Sample pitch data for two pitchers across one game."""
    return pl.LazyFrame({
        "game_pk": [1] * 10,
        "pitcher_id": [100, 100, 100, 100, 100, 200, 200, 200, 200, 200],
        "pitcher_name": ["Ace P"] * 5 + ["Relief P"] * 5,
        "batter_id": [300, 300, 300, 301, 301, 302, 302, 302, 303, 303],
        "batter_name": ["Bat A"] * 3 + ["Bat B"] * 2 + ["Bat C"] * 3 + ["Bat D"] * 2,
        "pitch_type": ["FF", "SL", "FF", "CU", "FF", "SI", "SL", "FF", "CH", "FF"],
        "start_speed": [95.0, 85.0, 96.0, 78.0, 94.0, 92.0, 84.0, 93.0, 82.0, 95.0],
        "is_strike": [True, True, False, False, True, True, False, True, True, False],
        "is_ball": [False, False, True, True, False, False, True, False, False, True],
        "is_in_play": [False, False, True, False, False, False, False, True, False, True],
        "call_code": ["C", "S", "X", "B", "S", "C", "B", "X", "S", "X"],
        "inning": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "half_inning": ["top"] * 5 + ["bottom"] * 5,
        "at_bat_index": [0, 0, 0, 1, 1, 2, 2, 2, 3, 3],
    })


@pytest.fixture
def at_bats_lf() -> pl.LazyFrame:
    """Sample at-bat outcomes matching the pitch data."""
    return pl.LazyFrame({
        "game_pk": [1, 1, 1, 1],
        "at_bat_index": [0, 1, 2, 3],
        "batter_id": [300, 301, 302, 303],
        "batter_name": ["Bat A", "Bat B", "Bat C", "Bat D"],
        "pitcher_id": [100, 100, 200, 200],
        "pitcher_name": ["Ace P", "Ace P", "Relief P", "Relief P"],
        "event": ["Single", "Strikeout", "Double", "Groundout"],
        "event_type": ["single", "strikeout", "double", "field_out"],
        "rbi": [1, 0, 2, 0],
        "half_inning": ["top", "top", "bottom", "bottom"],
        "is_scoring_play": [True, False, True, False],
        "inning": [1, 1, 2, 2],
    })


@pytest.fixture
def games_lf() -> pl.LazyFrame:
    return pl.LazyFrame({
        "game_pk": [1],
        "game_date": ["2026-03-28"],
        "away_team_name": ["NYY"],
        "home_team_name": ["SF"],
        "away_score": [1],
        "home_score": [2],
        "status": ["Final"],
    })


class TestAggregatePitcherGame:
    def test_produces_one_row_per_pitcher(self, pitches_lf):
        result = aggregate_pitcher_game(pitches_lf).collect()
        assert result.height == 2
        assert set(result["pitcher_id"].to_list()) == {100, 200}

    def test_total_pitches_correct(self, pitches_lf):
        result = aggregate_pitcher_game(pitches_lf).collect()
        ace = result.filter(pl.col("pitcher_id") == 100)
        assert ace["total_pitches"][0] == 5

    def test_strike_count(self, pitches_lf):
        result = aggregate_pitcher_game(pitches_lf).collect()
        ace = result.filter(pl.col("pitcher_id") == 100)
        assert ace["strikes"][0] == 3  # C, S, S

    def test_pitch_type_counts(self, pitches_lf):
        result = aggregate_pitcher_game(pitches_lf).collect()
        ace = result.filter(pl.col("pitcher_id") == 100)
        assert ace["ff_count"][0] == 3
        assert ace["sl_count"][0] == 1
        assert ace["cu_count"][0] == 1


class TestAggregateBatterGame:
    def test_produces_one_row_per_batter(self, at_bats_lf, pitches_lf):
        result = aggregate_batter_game(at_bats_lf, pitches_lf).collect()
        assert result.height == 4

    def test_hits_counted(self, at_bats_lf, pitches_lf):
        result = aggregate_batter_game(at_bats_lf, pitches_lf).collect()
        bat_a = result.filter(pl.col("batter_id") == 300)
        assert bat_a["hits"][0] == 1  # single

    def test_batting_avg(self, at_bats_lf, pitches_lf):
        result = aggregate_batter_game(at_bats_lf, pitches_lf).collect()
        bat_a = result.filter(pl.col("batter_id") == 300)
        assert bat_a["batting_avg"][0] == pytest.approx(1.0)  # 1 hit / 1 AB

    def test_rbi_summed(self, at_bats_lf, pitches_lf):
        result = aggregate_batter_game(at_bats_lf, pitches_lf).collect()
        bat_c = result.filter(pl.col("batter_id") == 302)
        assert bat_c["rbi"][0] == 2


class TestAggregateTeamGame:
    def test_two_rows_per_game(self, at_bats_lf):
        result = aggregate_team_game(at_bats_lf).collect()
        assert result.height == 2  # top (away) and bottom (home)

    def test_home_team_hits(self, at_bats_lf):
        result = aggregate_team_game(at_bats_lf).collect()
        home = result.filter(pl.col("half_inning") == "bottom")
        assert home["total_hits"][0] == 1  # double only


class TestRollingPlayerStats:
    def test_rolling_avg_computed(self):
        daily = pl.LazyFrame({
            "player_id": [1, 1, 1, 1, 1],
            "stat_date": ["2026-03-28", "2026-03-29", "2026-03-30", "2026-03-31", "2026-04-01"],
            "hits": [2, 1, 3, 0, 2],
            "at_bats_count": [4, 3, 4, 4, 4],
            "strikeouts": [1, 1, 0, 2, 1],
            "walks": [0, 1, 1, 0, 0],
            "plate_appearances": [4, 4, 5, 4, 4],
        })
        result = rolling_player_stats(daily, window=3).collect()
        assert "rolling_avg" in result.columns
        assert "rolling_k_rate" in result.columns
        assert "rolling_bb_rate" in result.columns
        assert "rolling_ops_approx" in result.columns
        assert result.height == 5

    def test_rolling_avg_values(self):
        daily = pl.LazyFrame({
            "player_id": [1, 1, 1],
            "stat_date": ["2026-03-28", "2026-03-29", "2026-03-30"],
            "hits": [1, 1, 1],
            "at_bats_count": [4, 4, 4],
            "strikeouts": [1, 1, 1],
            "walks": [0, 0, 0],
            "plate_appearances": [4, 4, 4],
        })
        result = rolling_player_stats(daily, window=3).collect()
        # After 3 games: 3 hits / 12 ABs = 0.25
        assert result["rolling_avg"][-1] == pytest.approx(0.25)


class TestGameSummary:
    def test_one_row_per_game(self, games_lf, pitches_lf, at_bats_lf):
        result = game_summary(games_lf, pitches_lf, at_bats_lf).collect()
        assert result.height == 1

    def test_winner_computed(self, games_lf, pitches_lf, at_bats_lf):
        result = game_summary(games_lf, pitches_lf, at_bats_lf).collect()
        assert result["winner"][0] == "SF"  # home_score 2 > away_score 1

    def test_totals_match(self, games_lf, pitches_lf, at_bats_lf):
        result = game_summary(games_lf, pitches_lf, at_bats_lf).collect()
        assert result["total_pitches"][0] == 10
        assert result["total_at_bats"][0] == 4
        assert result["total_hits"][0] == 2  # single + double
