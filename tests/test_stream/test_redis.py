"""Tests for Redis connectivity and stream operations."""

import json

import pytest
import redis as redis_lib

from mlb_pipeline.config import settings
from mlb_pipeline.models.events import PitchEvent
from mlb_pipeline.models.enums import HalfInning
from datetime import datetime


@pytest.fixture
def redis_client():
    r = redis_lib.from_url(settings.redis_url)
    yield r
    # Cleanup test streams
    for key in r.keys("mlb:test:*"):
        r.delete(key)
    r.close()


class TestRedisConnection:
    def test_ping(self, redis_client):
        assert redis_client.ping() is True

    def test_stream_write_read(self, redis_client):
        stream = "mlb:test:stream"
        redis_client.xadd(stream, {"data": "test_value"})
        length = redis_client.xlen(stream)
        assert length == 1

        messages = redis_client.xrange(stream)
        assert len(messages) == 1
        _, fields = messages[0]
        assert fields[b"data"] == b"test_value"

        redis_client.delete(stream)

    def test_stream_pitch_event_round_trip(self, redis_client):
        """Serialize a PitchEvent to stream and deserialize it back."""
        stream = "mlb:test:pitches"
        event = PitchEvent(
            game_pk=823243,
            event_id="823243_0_1",
            timestamp=datetime(2026, 3, 27, 20, 35, 0),
            inning=1,
            half_inning=HalfInning.TOP,
            at_bat_index=0,
            pitch_number=1,
            pitcher_id=548389,
            pitcher_name="Robbie Ray",
            batter_id=502671,
            batter_name="Paul Goldschmidt",
            pitch_type="SL",
            start_speed=85.3,
            spin_rate=2152,
            call_code="C",
            call_description="Called Strike",
            is_in_play=False,
            is_strike=True,
            is_ball=False,
            balls=0,
            strikes=1,
            outs=0,
        )

        # Write to stream
        redis_client.xadd(stream, {"data": event.model_dump_json()})

        # Read back
        messages = redis_client.xrange(stream)
        assert len(messages) == 1
        _, fields = messages[0]
        restored = PitchEvent.model_validate_json(fields[b"data"])

        assert restored.game_pk == 823243
        assert restored.pitcher_name == "Robbie Ray"
        assert restored.pitch_type == "SL"
        assert restored.start_speed == 85.3
        assert restored.is_strike is True

        redis_client.delete(stream)

    def test_consumer_group(self, redis_client):
        """Test consumer group creation and reading."""
        stream = "mlb:test:consumer"
        group = "test_group"
        consumer = "test_consumer_1"

        # Add some messages
        redis_client.xadd(stream, {"data": "msg1"})
        redis_client.xadd(stream, {"data": "msg2"})

        # Create consumer group
        redis_client.xgroup_create(stream, group, id="0")

        # Read as consumer
        messages = redis_client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: ">"},
            count=10,
        )
        assert len(messages) == 1  # One stream
        stream_name, entries = messages[0]
        assert len(entries) == 2  # Two messages

        # Acknowledge
        for msg_id, _ in entries:
            redis_client.xack(stream, group, msg_id)

        # Verify pending is empty
        pending = redis_client.xpending(stream, group)
        assert pending["pending"] == 0

        redis_client.delete(stream)
