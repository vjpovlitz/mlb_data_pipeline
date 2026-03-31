"""Tests for Polars pitch-level transforms."""

import polars as pl
import pytest

from mlb_pipeline.processing.transforms import (
    add_count_features,
    add_game_context,
    add_pitch_sequence_features,
    add_zone_features,
    enrich_pitches,
)


@pytest.fixture
def sample_pitches() -> pl.LazyFrame:
    """Minimal pitch data covering common scenarios."""
    return pl.LazyFrame({
        "game_pk": [1, 1, 1, 1, 1, 1],
        "at_bat_index": [0, 0, 0, 1, 1, 1],
        "pitch_number": [1, 2, 3, 1, 2, 3],
        "pitch_type": ["FF", "SL", "FF", "CU", "FF", "SL"],
        "start_speed": [95.0, 85.0, 96.0, 78.0, 94.0, 86.0],
        "plate_x": [0.1, -0.9, 0.5, 0.0, 1.2, -0.2],
        "plate_z": [2.5, 3.8, 1.8, 2.8, 0.5, 2.5],
        "inning": [1, 1, 1, 1, 1, 1],
        "half_inning": ["top", "top", "top", "top", "top", "top"],
        "balls": [0, 0, 0, 0, 1, 1],
        "strikes": [1, 2, 2, 0, 0, 1],
        "outs": [0, 0, 0, 0, 0, 0],
        "home_score": [0, 0, 0, 0, 0, 0],
        "away_score": [0, 0, 0, 0, 0, 0],
        "runner_on_first": [False, False, False, True, True, True],
        "runner_on_second": [False, False, False, False, True, True],
        "runner_on_third": [False, False, False, False, False, True],
        "is_in_play": [False, False, True, False, False, True],
        "is_strike": [True, True, False, False, True, False],
        "is_ball": [False, False, False, True, False, False],
        "call_code": ["C", "S", "X", "B", "S", "X"],
        "zone": [5, None, 6, 5, None, 5],
    })


class TestPitchSequenceFeatures:
    def test_adds_prev_columns(self, sample_pitches):
        result = add_pitch_sequence_features(sample_pitches).collect()
        assert "prev_pitch_type" in result.columns
        assert "prev_speed" in result.columns
        assert "speed_diff" in result.columns
        assert "pitch_number_in_ab" in result.columns

    def test_first_pitch_has_null_prev(self, sample_pitches):
        result = add_pitch_sequence_features(sample_pitches).collect()
        # First pitch of each at-bat should have null prev
        first_pitches = result.filter(pl.col("pitch_number") == 1)
        assert first_pitches["prev_pitch_type"].null_count() == first_pitches.height

    def test_speed_diff_correct(self, sample_pitches):
        result = add_pitch_sequence_features(sample_pitches).collect()
        # Second pitch of AB 0: SL at 85, prev FF at 95 → diff = -10
        row = result.filter(
            (pl.col("at_bat_index") == 0) & (pl.col("pitch_number") == 2)
        )
        assert row["speed_diff"][0] == pytest.approx(-10.0)

    def test_pitch_number_in_ab_sequential(self, sample_pitches):
        result = add_pitch_sequence_features(sample_pitches).collect()
        for ab_idx in [0, 1]:
            ab = result.filter(pl.col("at_bat_index") == ab_idx)
            assert ab["pitch_number_in_ab"].to_list() == [1, 2, 3]


class TestZoneFeatures:
    def test_adds_zone_columns(self, sample_pitches):
        result = add_zone_features(sample_pitches).collect()
        for col in ["is_in_zone", "zone_region", "horizontal_location", "vertical_location"]:
            assert col in result.columns

    def test_heart_of_zone(self, sample_pitches):
        result = add_zone_features(sample_pitches).collect()
        # plate_x=0.0, plate_z=2.8 → heart (inner third on both axes)
        row = result.filter(
            (pl.col("at_bat_index") == 1) & (pl.col("pitch_number") == 1)
        )
        assert row["zone_region"][0] == "heart"
        assert row["is_in_zone"][0] is True

    def test_chase_pitch(self, sample_pitches):
        result = add_zone_features(sample_pitches).collect()
        # plate_x=-0.9, plate_z=3.8 → outside zone but within chase buffer
        row = result.filter(
            (pl.col("at_bat_index") == 0) & (pl.col("pitch_number") == 2)
        )
        assert row["zone_region"][0] == "chase"
        assert row["is_in_zone"][0] is False

    def test_waste_pitch(self, sample_pitches):
        result = add_zone_features(sample_pitches).collect()
        # plate_x=1.2, plate_z=0.5 → far outside on both axes
        row = result.filter(
            (pl.col("at_bat_index") == 1) & (pl.col("pitch_number") == 2)
        )
        assert row["zone_region"][0] == "waste"


class TestGameContext:
    def test_adds_context_columns(self, sample_pitches):
        result = add_game_context(sample_pitches).collect()
        for col in ["score_diff", "is_close_game", "base_state", "runners_on_base",
                     "is_scoring_position", "leverage_bucket"]:
            assert col in result.columns

    def test_base_state_encoding(self, sample_pitches):
        result = add_game_context(sample_pitches).collect()
        # Row with runners on 1st+2nd+3rd: 1+2+4=7
        row = result.filter(
            (pl.col("at_bat_index") == 1) & (pl.col("pitch_number") == 3)
        )
        assert row["base_state"][0] == 7
        assert row["runners_on_base"][0] == 3

    def test_no_runners(self, sample_pitches):
        result = add_game_context(sample_pitches).collect()
        row = result.filter(
            (pl.col("at_bat_index") == 0) & (pl.col("pitch_number") == 1)
        )
        assert row["base_state"][0] == 0
        assert row["runners_on_base"][0] == 0

    def test_close_game_tie(self, sample_pitches):
        result = add_game_context(sample_pitches).collect()
        # All scores 0-0, should be close
        assert result["is_close_game"].all()


class TestCountFeatures:
    def test_adds_count_columns(self, sample_pitches):
        result = add_count_features(sample_pitches).collect()
        for col in ["is_first_pitch", "is_full_count", "is_hitter_count",
                     "is_pitcher_count", "count_label"]:
            assert col in result.columns

    def test_first_pitch_detected(self, sample_pitches):
        result = add_count_features(sample_pitches).collect()
        first = result.filter(pl.col("pitch_number") == 1)
        assert first["is_first_pitch"].all()

    def test_hitter_count(self, sample_pitches):
        result = add_count_features(sample_pitches).collect()
        # Ball 1, Strike 0 → hitter count
        row = result.filter(
            (pl.col("at_bat_index") == 1) & (pl.col("pitch_number") == 2)
        )
        assert row["is_hitter_count"][0] is True

    def test_pitcher_count(self, sample_pitches):
        result = add_count_features(sample_pitches).collect()
        # Ball 0, Strike 2 → pitcher count
        row = result.filter(
            (pl.col("at_bat_index") == 0) & (pl.col("pitch_number") == 2)
        )
        assert row["is_pitcher_count"][0] is True

    def test_count_label_format(self, sample_pitches):
        result = add_count_features(sample_pitches).collect()
        row = result.filter(
            (pl.col("at_bat_index") == 1) & (pl.col("pitch_number") == 3)
        )
        assert row["count_label"][0] == "1-1"


class TestEnrichPitches:
    def test_all_columns_present(self, sample_pitches):
        result = enrich_pitches(sample_pitches).collect()
        expected = {
            "prev_pitch_type", "prev_speed", "speed_diff", "pitch_number_in_ab",
            "is_in_zone", "zone_region", "horizontal_location", "vertical_location",
            "score_diff", "is_close_game", "base_state", "runners_on_base",
            "is_scoring_position", "leverage_bucket",
            "is_first_pitch", "is_full_count", "is_hitter_count",
            "is_pitcher_count", "count_label",
        }
        assert expected.issubset(set(result.columns))

    def test_row_count_preserved(self, sample_pitches):
        original = sample_pitches.collect().height
        enriched = enrich_pitches(sample_pitches).collect().height
        assert enriched == original
