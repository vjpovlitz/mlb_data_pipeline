"""Tests for parsing MLB live feed into pipeline events."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from mlb_pipeline.ingestion.parser import parse_pitch_events, parse_at_bat_result
from mlb_pipeline.models.events import AtBatResult, PitchEvent

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def live_feed():
    with open(FIXTURES_DIR / "live_feed_sample.json") as f:
        return json.load(f)


@pytest.fixture
def first_at_bat(live_feed):
    return live_feed["liveData"]["plays"]["allPlays"][0]


class TestParsePitchEvents:
    def test_parses_pitches_from_at_bat(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        pitches = parse_pitch_events(game_pk, first_at_bat)
        assert len(pitches) > 0
        assert all(isinstance(p, PitchEvent) for p in pitches)

    def test_skips_non_pitch_events(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        pitches = parse_pitch_events(game_pk, first_at_bat)
        # First AB has 3 non-pitch events + 4 pitches = 7 total events
        # Parser should only return the 4 pitches
        assert len(pitches) == 4

    def test_pitch_event_fields_populated(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        pitches = parse_pitch_events(game_pk, first_at_bat)
        pitch = pitches[0]

        assert pitch.game_pk == game_pk
        assert pitch.inning == 1
        assert pitch.pitcher_id > 0
        assert pitch.batter_id > 0
        assert pitch.pitcher_name != ""
        assert pitch.batter_name != ""
        assert pitch.pitch_type in ("SL", "FF", "CU", "CH", "SI", "FC", "KC", "FS", "ST", None)
        assert pitch.start_speed is not None
        assert 40 < pitch.start_speed < 110
        assert pitch.call_code in ("C", "S", "B", "F", "X", "T", "M", "D", "E", "L", "H", "P", "Q", "R", "V", "W", "Z")

    def test_pitch_numbers_sequential(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        pitches = parse_pitch_events(game_pk, first_at_bat)
        numbers = [p.pitch_number for p in pitches]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_event_ids_unique(self, live_feed):
        game_pk = live_feed["gamePk"]
        all_event_ids = set()
        for play in live_feed["liveData"]["plays"]["allPlays"]:
            pitches = parse_pitch_events(game_pk, play)
            for p in pitches:
                assert p.event_id not in all_event_ids, f"Duplicate event_id: {p.event_id}"
                all_event_ids.add(p.event_id)
        assert len(all_event_ids) > 50

    def test_count_state_valid(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        pitches = parse_pitch_events(game_pk, first_at_bat)
        for p in pitches:
            assert 0 <= p.balls <= 4
            assert 0 <= p.strikes <= 3
            assert 0 <= p.outs <= 3


class TestParseAtBatResult:
    def test_parses_at_bat_result(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        result = parse_at_bat_result(game_pk, first_at_bat)
        assert isinstance(result, AtBatResult)
        assert result.game_pk == game_pk
        assert result.event != ""
        assert result.event_type != ""
        assert result.description != ""

    def test_all_at_bats_parseable(self, live_feed):
        game_pk = live_feed["gamePk"]
        results = []
        for play in live_feed["liveData"]["plays"]["allPlays"]:
            result = parse_at_bat_result(game_pk, play)
            results.append(result)
        assert len(results) == 67  # Known count for this game

    def test_at_bat_pitch_count_matches(self, live_feed, first_at_bat):
        game_pk = live_feed["gamePk"]
        pitches = parse_pitch_events(game_pk, first_at_bat)
        result = parse_at_bat_result(game_pk, first_at_bat)
        assert result.pitch_count == len(pitches)

    def test_scoring_play_detection(self, live_feed):
        game_pk = live_feed["gamePk"]
        scoring_plays = []
        for play in live_feed["liveData"]["plays"]["allPlays"]:
            result = parse_at_bat_result(game_pk, play)
            if result.is_scoring_play:
                scoring_plays.append(result)
        # Yankees won 3-0, so at least 3 RBI worth of scoring plays
        total_rbi = sum(r.rbi for r in scoring_plays)
        assert total_rbi >= 3
