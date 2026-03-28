# MLB Pipeline — Testing Documentation

## Overview

The test suite validates all Phase 1 infrastructure: data models, API client, feed parsing, SQL Server connectivity, and Redis Streams operations. All tests run locally against real services (SQL Server, Redis) and recorded API fixtures.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=mlb_pipeline --cov-report=html

# Run specific test module
pytest tests/test_ingestion/test_feed_parser.py -v
```

## Prerequisites

- SQL Server running on localhost with `mlb_pipeline` database created
- Redis running on localhost:6379 (via `docker compose up -d`)
- Package installed: `pip install -e ".[dev]"`
- Migrations applied: `python -m mlb_pipeline.cli db migrate`
- Teams seeded: `python -m mlb_pipeline.cli db seed-teams`

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures (schedule, live feed JSON)
├── fixtures/
│   ├── schedule_sample.json             # Real MLB schedule (2026-03-28, 15 games)
│   └── live_feed_sample.json            # Complete game: NYY 3-0 SF (2026-03-27)
├── test_ingestion/
│   ├── test_models.py                   # Pydantic models, enums, config
│   ├── test_client.py                   # MLB Stats API client (mocked + fixture)
│   └── test_feed_parser.py              # Live feed → PitchEvent/AtBatResult parsing
├── test_storage/
│   └── test_sqlserver.py                # SQL Server connectivity and ORM round-trips
├── test_stream/
│   └── test_redis.py                    # Redis Streams write/read/consumer groups
├── test_processing/                     # (Phase 4 — feature engineering)
├── test_ml/                             # (Phase 4 — model training/inference)
└── integration/                         # (Phase 5 — end-to-end pipeline)
```

## Test Details

### test_ingestion/test_models.py (9 tests)

Tests the foundational data types that flow through the entire pipeline.

| Test | What it validates |
|------|-------------------|
| `TestSettings::test_default_settings` | Pydantic Settings loads defaults, connection string format |
| `TestSettings::test_connection_string_format` | SQL Server connection string includes DB name and trusted auth |
| `TestEnums::test_game_states` | GameState enum values match MLB API strings |
| `TestEnums::test_half_inning` | HalfInning.TOP/BOTTOM match API values |
| `TestEnums::test_pitch_types` | PitchType codes (FF, SL, CU, etc.) are correct |
| `TestPitchEvent::test_create_pitch_event` | Full PitchEvent construction with all fields |
| `TestPitchEvent::test_pitch_event_json_round_trip` | Serialize to JSON and deserialize back, equality check |
| `TestAtBatResult::test_create_at_bat_result` | AtBatResult construction and field validation |
| `TestGameStateEvent::test_create_game_state_event` | GameStateEvent with state transitions |

### test_ingestion/test_client.py (11 tests)

Tests the async MLB Stats API client using mocked HTTP sessions and recorded fixtures.

| Test | What it validates |
|------|-------------------|
| `test_get_schedule_parses_games` | Schedule API response has dates/games structure |
| `test_live_feed_has_required_structure` | Live feed fixture has gameData + liveData |
| `test_live_feed_play_structure` | Play entries have result, matchup, playEvents, about |
| `test_live_feed_pitch_event_structure` | Pitch events contain details, count, pitchData |
| `test_live_feed_game_completed` | Fixture game status is "Final" |
| `test_schedule_range_flattens_games` | get_schedule_range returns flat game list |
| `test_context_manager` | Async context manager lifecycle |
| `test_schedule_has_games` | Fixture data quality — games exist |
| `test_live_feed_has_pitches` | Fixture has 50+ pitches (actual: 275) |
| `test_live_feed_game_pk` | Game PK matches expected value (823243) |
| `test_live_feed_has_scores` | Linescore has integer run totals |

### test_ingestion/test_feed_parser.py (10 tests)

Tests the parser that converts raw MLB API JSON into canonical pipeline events. Uses the recorded Yankees vs Giants game (275 pitches, 67 at-bats).

| Test | What it validates |
|------|-------------------|
| `test_parses_pitches_from_at_bat` | Extracts PitchEvent list from a single at-bat |
| `test_skips_non_pitch_events` | Filters out pickoffs, mound visits, etc. (first AB: 7 events → 4 pitches) |
| `test_pitch_event_fields_populated` | Speed (40-110 mph), pitch type, call code all present |
| `test_pitch_numbers_sequential` | Pitch numbers are 1, 2, 3... within each at-bat |
| `test_event_ids_unique` | No duplicate event_id across entire game (275 unique) |
| `test_count_state_valid` | Balls 0-4, strikes 0-3, outs 0-3 |
| `test_parses_at_bat_result` | AtBatResult has event, event_type, description |
| `test_all_at_bats_parseable` | All 67 at-bats parse without error |
| `test_at_bat_pitch_count_matches` | AtBatResult.pitch_count equals len(parsed pitches) |
| `test_scoring_play_detection` | Finds scoring plays, total RBI >= 3 (actual: Judge 2-run HR + Stanton solo HR) |

### test_storage/test_sqlserver.py (4 tests)

Tests SQL Server connectivity and ORM operations against the live database.

| Test | What it validates |
|------|-------------------|
| `test_can_connect` | SELECT 1 succeeds via SQLAlchemy engine |
| `test_tables_exist` | All 7 tables present (games, pitches, at_bats, teams, players, player_stats_daily, win_probability_log) |
| `test_teams_seeded` | 30 MLB teams loaded in teams table |
| `test_game_write_read_delete` | Full ORM round-trip: insert Game → read back → verify fields → delete → confirm gone |

### test_stream/test_redis.py (4 tests)

Tests Redis Streams operations for the event buffering layer.

| Test | What it validates |
|------|-------------------|
| `test_ping` | Redis server is reachable |
| `test_stream_write_read` | XADD + XRANGE round-trip |
| `test_stream_pitch_event_round_trip` | Serialize PitchEvent to JSON → XADD → XRANGE → deserialize, verify all fields |
| `test_consumer_group` | XGROUP CREATE → XREADGROUP → XACK → verify zero pending |

## Test Fixture: Yankees vs Giants (2026-03-27)

The primary test fixture is a complete game captured from the MLB Stats API:

- **Game PK**: 823243
- **Matchup**: New York Yankees @ San Francisco Giants
- **Result**: Yankees 3, Giants 0
- **Pitches**: 275 total across 67 at-bats
- **Pitchers**: 11 (both teams combined)
- **Batters**: 20
- **Pitch types**: FF (90), SL (64), SI (62), FC (24), FS (11), CU (11), KC (7), CH (6)
- **Scoring plays**: 2 (Aaron Judge 2-run HR, Giancarlo Stanton solo HR — both in 6th inning)
- **File size**: ~836 KB

This fixture enables deterministic testing of the entire parsing pipeline without network calls.
