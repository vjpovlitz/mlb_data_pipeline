# MLB Real-Time Baseball Data Pipeline — Implementation Plan

## Context

Build a real-time MLB baseball data pipeline from scratch in `C:\Users\vjpov\Codebase\Sports` (currently empty). The pipeline ingests live game data from the free MLB Stats API, processes it through a streaming architecture, stores it in SQL Server, and feeds game outcome prediction models running on an RTX 5080 GPU. The primary development machine is a Windows Desktop (14-core CPU, NVIDIA RTX 5080, SQL Server installed, Docker Desktop available).

**Why**: Enable real-time win probability and game outcome predictions during live MLB games, backed by historical data for model training and backtesting.

---

## Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Data Source** | MLB Stats API (`statsapi.mlb.com`) | Free, no auth, pitch-by-pitch live feeds, 1-2ms latency |
| **Historical Data** | `pybaseball` + Statcast | Richest historical pitch dataset available (2015-present) |
| **Ingestion** | Python `asyncio` + `aiohttp` | Concurrent polling of multiple live games |
| **Message Buffer** | Redis Streams (Docker) | Ultra-low latency, bounded streams, consumer groups |
| **Processing** | Polars | 3-10x faster than Pandas, lazy eval, auto-parallelizes on 14 cores |
| **Relational DB** | SQL Server (existing) | Already installed, stores games/pitches/stats/predictions |
| **Analytics** | DuckDB over Parquet | Zero-config, blazing fast for feature engineering on historical data |
| **ML Framework** | PyTorch + CUDA (RTX 5080) | GPU-accelerated training and real-time inference |
| **Orchestration** | Prefect | Modern Python-native, simple local setup, async-first |
| **Dashboard** | Grafana (Docker) | Real-time game viz, pipeline health, prediction panels |
| **API Layer** | FastAPI | Bridges Grafana/clients to pipeline data |
| **Testing** | pytest + fakeredis + Great Expectations | Unit, integration, data quality |

---

## Project Structure

```
C:\Users\vjpov\Codebase\Sports\
├── pyproject.toml                     # Dependencies, project config
├── docker-compose.yml                 # Redis + Grafana only (SQL Server is native)
├── .env.example / .env                # Config (DB connection, Redis URL, etc.)
├── alembic.ini                        # DB migrations config
├── alembic/
│   ├── env.py
│   └── versions/                      # SQL Server migration scripts
├── src/
│   └── mlb_pipeline/
│       ├── __init__.py
│       ├── config.py                  # Pydantic Settings (SQL Server conn, Redis, paths)
│       ├── models/
│       │   ├── events.py              # Pydantic: PitchEvent, AtBatResult, GameStateEvent
│       │   ├── database.py            # SQLAlchemy ORM (SQL Server tables)
│       │   └── enums.py               # GameState, PitchType, HalfInning
│       ├── ingestion/
│       │   ├── client.py              # Async MLB Stats API client
│       │   ├── poller.py              # Live game polling + diff algorithm
│       │   ├── schedule.py            # Game schedule discovery
│       │   └── statcast.py            # pybaseball historical loader
│       ├── stream/
│       │   ├── producer.py            # Redis Streams publisher
│       │   ├── consumer.py            # Consumer group base class
│       │   └── handlers.py            # Event handlers (store, aggregate, predict)
│       ├── processing/
│       │   ├── transforms.py          # Polars pitch/game transforms
│       │   ├── features.py            # Win probability feature engineering
│       │   └── aggregations.py        # Rolling stats, game-level aggregations
│       ├── storage/
│       │   ├── sqlserver.py           # SQL Server read/write (async SQLAlchemy + aioodbc)
│       │   ├── duckdb_store.py        # DuckDB analytical queries over Parquet
│       │   └── parquet.py             # Parquet file partitioning/management
│       ├── ml/
│       │   ├── dataset.py             # PyTorch Dataset for win probability
│       │   ├── models.py              # LSTM + game state encoder architecture
│       │   ├── trainer.py             # Training loop (CUDA on RTX 5080)
│       │   ├── inference.py           # Real-time win probability service
│       │   └── evaluation.py          # Model eval: Brier score, calibration, accuracy
│       ├── quality/
│       │   ├── validators.py          # Data quality rules (Great Expectations)
│       │   └── monitors.py            # Pipeline health checks
│       ├── dashboard/
│       │   ├── api.py                 # FastAPI endpoints
│       │   └── provisioning/          # Grafana dashboard JSONs
│       └── cli.py                     # CLI entry points (click)
├── tests/
│   ├── conftest.py                    # Fixtures (fakeredis, test DB, sample data)
│   ├── fixtures/                      # Recorded API responses, sample Parquet
│   ├── test_ingestion/
│   ├── test_stream/
│   ├── test_processing/
│   ├── test_storage/
│   ├── test_ml/
│   └── integration/
├── scripts/
│   ├── seed_historical.py             # Bulk historical data loader
│   ├── replay_game.py                 # Replay recorded games for testing
│   └── export_training_data.py        # Export features to Parquet for ML
└── data/                              # Local data (gitignored)
    ├── parquet/
    ├── models/
    └── cache/
```

---

## Phase 1: Core Infrastructure

**Goal**: Project skeleton, Docker services (Redis + Grafana), SQL Server schema, basic API connectivity proven.

### Steps

1. **Create `pyproject.toml`** — PEP 621 with `src` layout. Core deps: `aiohttp`, `httpx`, `redis`, `sqlalchemy[asyncio]`, `aioodbc`, `pyodbc`, `alembic`, `pydantic`, `pydantic-settings`, `polars`, `duckdb`, `pyarrow`, `MLB-StatsAPI`, `pybaseball`, `click`, `structlog`, `tenacity`. Optional groups: `[ml]` (torch), `[quality]` (great-expectations), `[dashboard]` (fastapi, uvicorn), `[dev]` (pytest, fakeredis, ruff, mypy).

2. **Create `docker-compose.yml`** — Two services only:
   - **Redis** (`redis:7-alpine`): port 6379, AOF persistence
   - **Grafana** (`grafana/grafana:11`): port 3000, provisioned datasources
   - SQL Server is native on the machine, not in Docker.

3. **Create `.env.example` and `src/mlb_pipeline/config.py`** — Pydantic Settings loading from `.env`. SQL Server connection string via `pyodbc`/`aioodbc`. Key settings: `db_connection_string`, `redis_url`, `poll_interval_live` (2s), `poll_interval_idle` (60s), `data_dir`.

4. **Create `src/mlb_pipeline/models/enums.py`** — `GameState`, `PitchType`, `HalfInning`, `EventType` enums.

5. **Create `src/mlb_pipeline/models/events.py`** — Pydantic v2 models:
   - `PitchEvent` — Single pitch: game_pk, event_id, timestamp, inning, pitcher/batter IDs, pitch metrics (speed, spin, location), call, hit data, count state
   - `AtBatResult` — Completed at-bat: event type, RBI, scores, is_scoring_play
   - `GameStateEvent` — State transitions (Preview→Live→Final)

6. **Create `src/mlb_pipeline/models/database.py`** — SQLAlchemy 2.0 ORM for SQL Server:
   - `games` — game_pk (PK), date, teams, venue, status, scores, game_type, season
   - `pitches` — (game_pk, at_bat_index, pitch_number) composite PK, all pitch metrics, indexed on (game_pk), (pitcher_id, timestamp), (batter_id, timestamp)
   - `at_bats` — at-bat results, indexed on game_pk
   - `player_stats_daily` — aggregated daily stats per player
   - `teams`, `players` — dimension/reference tables
   - `win_probability_log` — stores real-time predictions per game state

7. **Create Alembic migration** — Configure for SQL Server dialect (`mssql+pyodbc`). Initial migration creates all tables and indexes.

8. **Create `src/mlb_pipeline/ingestion/client.py`** — Async HTTP client:
   - `get_schedule(date)` → `/api/v1/schedule`
   - `get_live_feed(game_pk)` → `/api/v1.1/game/{gamePk}/feed/live`
   - `get_play_by_play(game_pk)`, `get_boxscore(game_pk)`
   - Connection pooling via single `aiohttp.ClientSession`, retry with `tenacity`

9. **Create `src/mlb_pipeline/cli.py`** — Click CLI: `mlb ingest live`, `mlb ingest backfill`, `mlb replay <game_pk>`, `mlb db migrate`, `mlb ml train`, `mlb ml evaluate`, `mlb dashboard start`

10. **Create `.gitignore`** — Python, data/, .env, __pycache__, *.pyc, .venv, data/

### Verify
- `docker compose up -d` → Redis + Grafana healthy
- `pip install -e ".[dev]"` → no errors
- `mlb db migrate` → tables created in SQL Server
- `python -c "from mlb_pipeline.ingestion.client import MLBStatsClient; ..."` → returns today's schedule
- `pytest tests/` → config loading and model validation pass

---

## Phase 2: Real-Time Pipeline

**Goal**: Poll live games, diff for new pitches/events, publish to Redis Streams, consume and store in SQL Server.

### Steps

1. **Create `src/mlb_pipeline/ingestion/poller.py`** — `LiveGamePoller`:
   - Main loop: discover active games → poll each concurrently → sleep
   - **Diff algorithm** (critical): Track `GameSnapshot` per game (last_play_index, event count per play, game state). Compare new feed vs snapshot to detect new pitches, completed at-bats, state changes. Handles: missed polls (processes all intermediate events), rain delays, reviews, mid-at-bat substitutions.
   - Emits `PitchEvent`, `AtBatResult`, `GameStateEvent` to producer

2. **Create `src/mlb_pipeline/ingestion/schedule.py`** — `ScheduleWatcher`: discovers today's games, returns live game_pks

3. **Create `src/mlb_pipeline/stream/producer.py`** — `RedisStreamProducer`:
   - Three bounded streams: `mlb:pitches`, `mlb:at_bats`, `mlb:game_state`
   - `XADD` with `MAXLEN ~100000`, JSON serialization via Pydantic

4. **Create `src/mlb_pipeline/stream/consumer.py`** — `RedisStreamConsumer` base:
   - Consumer groups via `XREADGROUP`, blocking reads, `XACK` after processing

5. **Create `src/mlb_pipeline/stream/handlers.py`** — Concrete handlers:
   - `PitchStorageHandler` → deserialize, write to SQL Server `pitches`
   - `AtBatStorageHandler` → write to `at_bats`
   - `GameStateHandler` → update `games` status, trigger end-of-game aggregation

6. **Create `src/mlb_pipeline/storage/sqlserver.py`** — `SQLServerStore`:
   - Async via `aioodbc` or sync via `pyodbc` with thread pool
   - `insert_pitch`, `insert_pitches_batch`, `insert_at_bat`, `update_game_state`
   - `get_pitcher_pitches`, `get_game_pitches` for downstream queries
   - Bulk insert using `executemany` for batch efficiency

### Verify
- Start poller during a live game: `mlb ingest live`
- Logs show pitches detected and published
- `redis-cli XLEN mlb:pitches` → count increasing
- SQL Server: `SELECT COUNT(*) FROM pitches WHERE game_pk = <game>` → rows accumulating
- No duplicate pitches (unique composite key constraint)
- `pytest tests/test_ingestion/ tests/test_stream/` passes with fakeredis + fixtures

---

## Phase 3: Historical Data & Game Replay

**Goal**: Backfill historical pitch data, enable game replay for testing without live games.

### Steps

1. **Create `src/mlb_pipeline/ingestion/statcast.py`** — `StatcastLoader`:
   - Uses `pybaseball.statcast()` in thread executor (it's sync)
   - Chunks requests into 5-day windows (25K row limit per query)
   - Saves to partitioned Parquet: `data/parquet/statcast/YYYY/MM/DD.parquet`
   - Optional load into SQL Server

2. **Create `src/mlb_pipeline/ingestion/historical.py`** — `HistoricalGameLoader`:
   - Loads completed games from MLB Stats API
   - Processes in batches of 10 concurrent requests
   - Extracts all events using same parsing logic as live poller

3. **Create `scripts/replay_game.py`** — `GameReplayer`:
   - Loads a completed game's pitch data from SQL Server or Parquet
   - Publishes events to Redis Streams with simulated timing (configurable speed: 10x, 50x, 100x)
   - Pipeline cannot distinguish replay from live — same streams, same consumers
   - **Critical for all subsequent development** — no need to wait for live games

4. **Create `src/mlb_pipeline/storage/duckdb_store.py`** — `DuckDBStore`:
   - Creates views over Parquet files: `read_parquet('data/parquet/statcast/**/*.parquet')`
   - Analytical queries: `pitcher_arsenal()`, `matchup_history()`, `season_stats()`
   - Returns Polars DataFrames via Arrow interchange

5. **Create `src/mlb_pipeline/storage/parquet.py`** — `ParquetManager`:
   - Partitioned writes: `write_daily_partition(df, dataset, date)`
   - Monthly compaction for query efficiency

### Verify
- `mlb ingest backfill --start 2025-06-01 --end 2025-06-07` → loads a week of data
- `ls data/parquet/statcast/2025/06/` → daily Parquet files exist
- DuckDB query over Parquet returns correct row counts
- `mlb replay 823243 --speed 50` → replays game, SQL Server rows appear
- `pytest tests/test_storage/` passes

---

## Phase 4: Feature Engineering & Win Probability Model

**Goal**: Build game-state features, train a win probability model on the RTX 5080, serve real-time predictions during live games.

### Steps

1. **Create `src/mlb_pipeline/processing/transforms.py`** — Polars transforms:
   - `add_pitch_sequence_features()` — prev pitch type, speed diff, location delta
   - `add_zone_features()` — pitch location classification
   - `add_game_context()` — score diff, inning, base-out state encoding
   - All use Polars lazy `LazyFrame` → composable, optimized query plans

2. **Create `src/mlb_pipeline/processing/features.py`** — `WinProbabilityFeatureEngineer`:
   - **Game state features**: inning, half, outs, base runners (8 states), score differential, home/away
   - **Momentum features**: runs scored in last N innings, recent pitching performance
   - **Historical context**: team win% in similar situations (from DuckDB), pitcher/batter quality metrics
   - **Pitch-level aggregation**: pitcher's current game stats (pitch count, K rate, BB rate so far)
   - Output: feature matrix where each row = one game state (after each at-bat), target = did home team win (binary)

3. **Create `src/mlb_pipeline/processing/aggregations.py`** — `Aggregator`:
   - `aggregate_game(game_pk)` — per-player game stats after game ends
   - `compute_daily_stats(date)` — daily rolling player stats
   - `compute_win_probability_features(game_pk)` — extract state snapshots for training

4. **Create `scripts/export_training_data.py`** — Export feature matrices:
   - Query historical games from SQL Server / Parquet
   - Run through feature engineering
   - Save to `data/parquet/features/win_prob_YYYY.parquet`

5. **Create `src/mlb_pipeline/ml/dataset.py`** — `WinProbabilityDataset(Dataset)`:
   - Each sample: game state features → binary label (home team win)
   - Multiple samples per game (one per at-bat = ~70 per game)
   - Train/val/test split by date (temporal, not random)

6. **Create `src/mlb_pipeline/ml/models.py`** — Two architectures:
   - `WinProbabilityMLP` — Baseline: Input → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Linear(1) → Sigmoid. Fast to train, easy to debug.
   - `WinProbabilityLSTM` — Sequential model that processes the game as a sequence of states. Captures momentum and trajectory. Input: sequence of game state vectors → LSTM(hidden=128) → Linear(1) → Sigmoid.

7. **Create `src/mlb_pipeline/ml/trainer.py`** — `WinProbabilityTrainer`:
   - Auto-detects CUDA (RTX 5080) / MPS (Mac) / CPU
   - Binary cross-entropy loss, AdamW optimizer, ReduceLROnPlateau scheduler
   - Early stopping on validation Brier score
   - Saves checkpoints to `data/models/`
   - Training ~2M samples (30K games × 70 states) should take ~10-15 min on RTX 5080

8. **Create `src/mlb_pipeline/ml/inference.py`** — `WinProbabilityService`:
   - Loads trained model, keeps on GPU
   - `predict_win_probability(game_pk)` → builds live features from current game state → returns P(home_win)
   - Integrated as a Redis stream consumer — triggers after each `AtBatResult`
   - Writes predictions to `win_probability_log` table in SQL Server

9. **Create `src/mlb_pipeline/ml/evaluation.py`** — `ModelEvaluator`:
   - **Brier score** — primary metric for probability calibration
   - **Calibration curve** — binned predicted vs actual win rates
   - **Accuracy by game phase** — early (1-3), mid (4-6), late (7-9), extras
   - **Accuracy by score differential** — close games vs blowouts
   - **Log loss** — overall probability quality
   - HTML report generation with charts

### Verify
- `python scripts/export_training_data.py --season 2025` → feature Parquet created
- `mlb ml train --data data/parquet/features/win_prob_2025.parquet --epochs 30` → GPU utilized (`nvidia-smi` shows usage), model saved
- `mlb ml evaluate --model data/models/win_prob_v1.pt` → prints Brier score, calibration, per-phase accuracy
- During `mlb replay`, predictions logged: compare predicted P(home_win) vs actual outcome
- `pytest tests/test_ml/ tests/test_processing/` passes

---

## Phase 5: Testing, Quality & Monitoring

**Goal**: Comprehensive test suite, data quality validation, pipeline health monitoring.

### Steps

1. **Create `tests/conftest.py`** — Shared fixtures:
   - `fake_redis` — fakeredis for stream tests
   - `sample_live_feed` — recorded MLB API JSON from `tests/fixtures/`
   - `sample_pitches_df` — Polars DataFrame from sample Parquet
   - Test DB fixture (SQL Server test database or SQLite for unit tests)

2. **Record test fixtures** during Phase 2 development:
   - `tests/fixtures/live_feed_sample.json` — complete game
   - `tests/fixtures/live_feed_partial.json` — mid-game (for diff tests)
   - `tests/fixtures/schedule_sample.json`
   - `tests/fixtures/statcast_sample.parquet` — 10K rows

3. **Write critical tests**:
   - `test_poller.py` — Diff algorithm edge cases: new pitch in same AB, new AB, game state change, missed polls (jump innings), no change → no events, rain delay, review/challenge
   - `test_producer.py` / `test_consumer.py` — Stream routing, serialization, ack
   - `test_transforms.py` / `test_features.py` — Correct columns, values, shapes
   - `test_inference.py` — Predictions in [0,1], sum to valid probabilities
   - `integration/test_pipeline_e2e.py` — Full flow: publish known game → consume → store → verify row counts

4. **Create `src/mlb_pipeline/quality/validators.py`** — Data quality rules:
   - Per-pitch: speed 50-110mph, plate_x ±3.0, plate_z -1.0 to 6.0, spin_rate 0-4000
   - Per-batch: no duplicate composite keys, timestamp ordering, monotonic count changes, known pitch types, NULL rate < 5%

5. **Create `src/mlb_pipeline/quality/monitors.py`** — `PipelineMonitor`:
   - `check_stream_lag()` — Redis consumer group lag
   - `check_data_freshness()` — time since last stored pitch per active game
   - `check_ingestion_rate()` — pitches/min over last 5 min

### Verify
- `pytest tests/ --cov=mlb_pipeline` → >80% coverage on ingestion/ and stream/
- `mlb quality validate --date 2025-06-15` → prints data quality report
- All poller diff edge cases pass

---

## Phase 6: Dashboard & Visualization

**Goal**: Grafana dashboards for live games, predictions, and pipeline health.

### Steps

1. **Create `src/mlb_pipeline/dashboard/api.py`** — FastAPI:
   - `GET /api/games/live` — current live games with scores
   - `GET /api/games/{game_pk}/pitches` — pitch data, optionally since timestamp
   - `GET /api/games/{game_pk}/win-probability` — time series of P(home_win) through the game
   - `GET /api/pipeline/health` — stream lag, freshness, ingestion rate
   - `GET /api/players/{player_id}/stats` — rolling player stats

2. **Create Grafana provisioning** — Auto-configured datasources:
   - SQL Server via ODBC datasource plugin
   - JSON API datasource → FastAPI

3. **Create dashboard JSONs**:
   - **Live Game** — Win probability line chart (updates per at-bat), pitch velocity time series, strike zone scatter, pitch mix pie, score/count/outs display
   - **Pipeline Health** — Stream lengths, consumer lag, pitches/min, data freshness, error rates
   - **Game Analytics** — Historical win probability accuracy, model calibration curve, per-phase performance

### Verify
- `mlb dashboard start` → FastAPI on port 8000
- `curl localhost:8000/api/pipeline/health` → JSON response
- Grafana at `localhost:3000` → dashboards load
- During `mlb replay`, win probability chart updates in near-real-time

---

## Key Architecture Decisions

1. **SQL Server over TimescaleDB** — Already installed, no extra Docker container, sufficient for this workload. Indexed queries on `(game_pk)` and `(pitcher_id, timestamp)` handle the access patterns. DuckDB handles analytical workloads over Parquet.

2. **Polling with diff (not full-feed storage)** — MLB live feed returns ~2MB per game per poll. Diffing extracts only new events (~200 bytes each), reducing storage by 99.97%.

3. **Redis Streams as buffer (not Kafka)** — Kafka is overkill for a personal project. Redis Streams provides consumer groups, bounded streams, and sub-millisecond latency with minimal ops.

4. **Game replay as Phase 3 (not Phase 5)** — Every subsequent phase depends on testing with realistic data flow. Replayer publishes to the same Redis Streams as live poller — pipeline can't distinguish replay from live.

5. **Win probability over pitch prediction** — Game outcome prediction is more impactful (one continuous prediction per game), better calibratable (binary outcome), and naturally consumes all the same underlying pitch data.

6. **Polars everywhere** — Lazy evaluation compiles into optimized plans. `over()` window functions are perfect for baseball analytics (partition by game, AB, pitcher). Auto-parallelizes across all 14 cores.

---

## Phase Sequencing

```
Phase 1 (Core) ──→ Phase 2 (Real-Time) ──→ Phase 3 (Historical + Replay)
                                                      │
                                           ┌──────────┴──────────┐
                                           ▼                      ▼
                                    Phase 4 (ML)           Phase 5 (Testing)
                                           │                      │
                                           └──────────┬──────────┘
                                                      ▼
                                              Phase 6 (Dashboard)
```

Phases 1→2→3 are sequential. Phases 4 and 5 can be partially parallelized after Phase 3. Phase 6 can start basic dashboards after Phase 2, completed after Phase 4.

---

## Critical Files (highest implementation complexity)

- `src/mlb_pipeline/ingestion/poller.py` — Diff algorithm must handle all edge cases without duplicates or missed events
- `src/mlb_pipeline/models/events.py` — Canonical data contract; every component depends on these shapes
- `src/mlb_pipeline/processing/features.py` — Feature quality directly determines model accuracy
- `src/mlb_pipeline/ml/models.py` — Win probability architecture choices
- `docker-compose.yml` — Infrastructure foundation (Redis + Grafana)
