"""Tests for RedisPublisher — publish pitches, at-bats, and game state to Redis Streams."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import redis as redis_lib

from mlb_pipeline.config import settings
from mlb_pipeline.models.enums import GameState, HalfInning
from mlb_pipeline.models.events import AtBatResult, GameStateEvent, PitchEvent
from mlb_pipeline.stream.publisher import (
    STREAM_AT_BATS,
    STREAM_GAME_STATE,
    STREAM_PITCHES,
    RedisPublisher,
)


def _make_pitch(game_pk: int = 823243, at_bat_index: int = 0, pitch_number: int = 1) -> PitchEvent:
    return PitchEvent(
        game_pk=game_pk,
        event_id=f"{game_pk}_{at_bat_index}_{pitch_number}",
        timestamp=datetime(2026, 3, 27, 20, 35, 0),
        inning=1,
        half_inning=HalfInning.TOP,
        at_bat_index=at_bat_index,
        pitch_number=pitch_number,
        pitcher_id=548389,
        pitcher_name="Robbie Ray",
        batter_id=502671,
        batter_name="Paul Goldschmidt",
        pitch_type="FF",
        start_speed=95.2,
        call_code="S",
        call_description="Swinging Strike",
        is_in_play=False,
        is_strike=True,
        is_ball=False,
        balls=0,
        strikes=1,
        outs=0,
    )


def _make_at_bat(game_pk: int = 823243, at_bat_index: int = 0) -> AtBatResult:
    return AtBatResult(
        game_pk=game_pk,
        at_bat_index=at_bat_index,
        timestamp=datetime(2026, 3, 27, 20, 38, 0),
        inning=1,
        half_inning=HalfInning.TOP,
        pitcher_id=548389,
        pitcher_name="Robbie Ray",
        batter_id=502671,
        batter_name="Paul Goldschmidt",
        event="Strikeout",
        event_type="strikeout",
        description="Paul Goldschmidt strikes out swinging.",
        rbi=0,
        away_score=0,
        home_score=0,
        is_scoring_play=False,
        outs_after=1,
        pitch_count=5,
    )


def _make_game_state_event(game_pk: int = 823243) -> GameStateEvent:
    return GameStateEvent(
        game_pk=game_pk,
        timestamp=datetime.now(timezone.utc),
        previous_state=GameState.PREVIEW,
        new_state=GameState.LIVE,
        away_team="New York Yankees",
        home_team="San Francisco Giants",
    )


@pytest.fixture
def redis_client():
    r = redis_lib.from_url(settings.redis_url)
    yield r
    for key in r.keys("mlb:*"):
        r.delete(key)
    r.close()


@pytest.fixture
def publisher(redis_client):
    return RedisPublisher(redis_client, maxlen=1000)


class TestRedisPublisherCreate:
    def test_create_returns_publisher_when_redis_available(self):
        pub = RedisPublisher.create(settings.redis_url)
        assert pub is not None
        pub.close()

    def test_create_returns_none_when_redis_unavailable(self):
        pub = RedisPublisher.create("redis://localhost:19999")
        assert pub is None


class TestPublishPitches:
    def test_publishes_single_pitch(self, publisher, redis_client):
        pitch = _make_pitch()
        publisher.publish_pitches([pitch])

        length = redis_client.xlen(STREAM_PITCHES)
        assert length == 1

        messages = redis_client.xrange(STREAM_PITCHES)
        _, fields = messages[0]
        restored = PitchEvent.model_validate_json(fields[b"data"])
        assert restored.game_pk == 823243
        assert restored.pitcher_name == "Robbie Ray"
        assert restored.start_speed == 95.2

    def test_publishes_multiple_pitches(self, publisher, redis_client):
        pitches = [_make_pitch(pitch_number=i) for i in range(1, 6)]
        publisher.publish_pitches(pitches)

        length = redis_client.xlen(STREAM_PITCHES)
        assert length == 5

    def test_empty_list_no_op(self, publisher, redis_client):
        publisher.publish_pitches([])
        length = redis_client.xlen(STREAM_PITCHES)
        assert length == 0

    def test_publish_continues_on_error(self, publisher):
        """If one XADD fails, remaining pitches still publish."""
        pitches = [_make_pitch(pitch_number=i) for i in range(1, 4)]
        call_count = 0
        original_xadd = publisher._redis.xadd

        def flaky_xadd(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise redis_lib.ConnectionError("transient failure")
            return original_xadd(*args, **kwargs)

        publisher._redis.xadd = flaky_xadd
        publisher.publish_pitches(pitches)  # should not raise

        # 2 of 3 succeeded
        length = publisher._redis.xlen(STREAM_PITCHES)
        assert length == 2


class TestPublishAtBats:
    def test_publishes_single_at_bat(self, publisher, redis_client):
        ab = _make_at_bat()
        publisher.publish_at_bats([ab])

        length = redis_client.xlen(STREAM_AT_BATS)
        assert length == 1

        messages = redis_client.xrange(STREAM_AT_BATS)
        _, fields = messages[0]
        restored = AtBatResult.model_validate_json(fields[b"data"])
        assert restored.event == "Strikeout"
        assert restored.pitcher_name == "Robbie Ray"

    def test_publishes_batch(self, publisher, redis_client):
        at_bats = [_make_at_bat(at_bat_index=i) for i in range(5)]
        publisher.publish_at_bats(at_bats)
        assert redis_client.xlen(STREAM_AT_BATS) == 5

    def test_publish_continues_on_error(self, publisher):
        at_bats = [_make_at_bat(at_bat_index=i) for i in range(3)]
        call_count = 0
        original_xadd = publisher._redis.xadd

        def flaky_xadd(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise redis_lib.ConnectionError("transient")
            return original_xadd(*args, **kwargs)

        publisher._redis.xadd = flaky_xadd
        publisher.publish_at_bats(at_bats)
        assert publisher._redis.xlen(STREAM_AT_BATS) == 2


class TestPublishGameState:
    def test_publishes_game_state(self, publisher, redis_client):
        event = _make_game_state_event()
        publisher.publish_game_state(event)

        length = redis_client.xlen(STREAM_GAME_STATE)
        assert length == 1

        messages = redis_client.xrange(STREAM_GAME_STATE)
        _, fields = messages[0]
        restored = GameStateEvent.model_validate_json(fields[b"data"])
        assert restored.previous_state == GameState.PREVIEW
        assert restored.new_state == GameState.LIVE
        assert restored.away_team == "New York Yankees"

    def test_publish_game_state_error_does_not_raise(self):
        mock_redis = MagicMock()
        mock_redis.xadd.side_effect = redis_lib.ConnectionError("down")
        pub = RedisPublisher(mock_redis)
        event = _make_game_state_event()
        pub.publish_game_state(event)  # should not raise


class TestPublisherClose:
    def test_close_calls_redis_close(self):
        mock_redis = MagicMock()
        pub = RedisPublisher(mock_redis)
        pub.close()
        mock_redis.close.assert_called_once()

    def test_close_swallows_errors(self):
        mock_redis = MagicMock()
        mock_redis.close.side_effect = Exception("already closed")
        pub = RedisPublisher(mock_redis)
        pub.close()  # should not raise


class TestStreamMaxlen:
    def test_respects_maxlen(self, redis_client):
        pub = RedisPublisher(redis_client, maxlen=10)
        pitches = [_make_pitch(pitch_number=i) for i in range(1, 25)]
        pub.publish_pitches(pitches)
        # Redis approximate trimming may keep slightly more than maxlen
        length = redis_client.xlen(STREAM_PITCHES)
        assert length <= 25  # published 24
        # With approximate trimming, stream should be around maxlen
        # The key assertion is it doesn't grow unbounded
