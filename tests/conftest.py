"""Shared test fixtures."""

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_schedule():
    """Sample MLB schedule API response."""
    path = FIXTURES_DIR / "schedule_sample.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"dates": []}


@pytest.fixture
def sample_live_feed():
    """Sample MLB live feed API response."""
    path = FIXTURES_DIR / "live_feed_sample.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}
