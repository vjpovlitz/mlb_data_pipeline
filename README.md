# MLB Real-Time Data Pipeline

End-to-end baseball data pipeline that ingests live game data from the MLB Stats API, stores it in SQL Server, streams events through Redis, and serves analytics via DuckDB over partitioned Parquet files. Orchestrated with Apache Airflow.

## Architecture

```
MLB Stats API  -->  LiveGamePoller (async)  -->  SQL Server (relational store)
                                            -->  Redis Streams (event fan-out)
                                            -->  Parquet (date-partitioned)
                                                    |
                                                 DuckDB (analytical queries)
                                                    |
                                              Polars transforms & aggregations
```

## Tech Stack

| Layer           | Technology                          |
|-----------------|-------------------------------------|
| Ingestion       | aiohttp, MLB-StatsAPI, pybaseball   |
| Relational DB   | SQL Server 2022 + SQLAlchemy 2.0    |
| Streaming       | Redis Streams (Docker)              |
| File Storage    | Parquet (Polars + PyArrow)          |
| Analytics       | DuckDB (in-memory over Parquet)     |
| Transforms      | Polars LazyFrames                   |
| Orchestration   | Apache Airflow 2.10 (Docker)        |
| Migrations      | Alembic                             |
| ML (planned)    | PyTorch + CUDA                      |

## Project Structure

```
src/mlb_pipeline/
  ingestion/      # MLB API client, feed parser, live game poller
  models/         # Pydantic v2 models, SQLAlchemy ORM, enums
  processing/     # Polars transforms (pitch sequencing, zone features,
                  #   game context) and aggregations (pitcher/batter/team)
  storage/        # SQL Server writes, Parquet export, DuckDB queries
  stream/         # Redis Streams publisher
  cli.py          # Click CLI (ingest, db, api-test commands)
  config.py       # pydantic-settings configuration

dags/             # Airflow DAGs (backfill, aggregations, quality, export)
tests/            # 142 tests — unit, integration, storage, streaming
```

## Prerequisites

- Python 3.12+
- SQL Server (local or remote) with ODBC Driver 17/18
- Docker Desktop (for Redis, Airflow, Grafana)

## Setup

```bash
# Install pipeline + dev dependencies
pip install -e ".[dev]"

# Start Docker services (Redis, Airflow, Grafana)
docker compose up -d

# Run database migrations
alembic upgrade head

# Seed team reference data
mlb db seed-teams

# Verify connectivity
mlb db test-connection
mlb api-test
```

## Airflow

Airflow runs entirely in Docker (avoids SQLAlchemy version conflicts with the pipeline).

```bash
docker compose up -d
```

Open http://localhost:8080 (login: `admin` / `airflow`).

**DAGs:**

| DAG                   | Schedule       | Description                                      |
|-----------------------|----------------|--------------------------------------------------|
| `daily_backfill`      | 8 AM ET daily  | Fetch completed games from MLB API, store to SQL Server, export to Parquet |
| `daily_aggregations`  | 9 AM ET daily  | Compute pitcher/batter game-level stats, update daily player stats |
| `data_quality`        | 10 AM ET daily | Validate pitch ranges, check duplicates, verify game completeness |
| `data_export`         | Monday weekly  | Export weekly data to Parquet, compact monthly, enrich pitches |

## CLI Commands

```bash
# Start live game polling
mlb ingest live

# Database operations
mlb db seed-teams
mlb db test-connection

# Test MLB API connectivity
mlb api-test
```

## Analytics Queries (DuckDB)

The `DuckDBStore` runs analytical SQL over date-partitioned Parquet files:

- **Pitcher Arsenal** — pitch mix, velocity, spin, whiff/zone rates
- **Matchup History** — head-to-head pitcher vs batter stats
- **Team Standings** — win/loss records, run differential
- **Game Pace** — per-half-inning pitch timing
- **Daily Pitch Leaders** — top pitchers by strikeouts
- **Season Batting Leaders** — batting avg, slugging, HR, RBI

## Polars Transforms

All transforms are pure functions on `pl.LazyFrame`:

- **Pitch sequencing** — previous pitch type, speed differential
- **Zone features** — heart/edge/chase/waste classification
- **Game context** — score differential, base state, leverage
- **Count features** — hitter's count, pitcher's count, full count

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=mlb_pipeline --cov-report=term-missing

# Specific module
pytest tests/test_processing/ -v
```

142 tests covering ingestion, parsing, transforms, aggregations, SQL Server writes, Parquet export, DuckDB queries, and Redis streaming.

## Data Layout (Parquet)

```
data/parquet/
  pitches/{YYYY-MM-DD}/{game_pk}.parquet
  at_bats/{YYYY-MM-DD}/{game_pk}.parquet
  games/{season}/games.parquet
  daily_stats/{YYYY-MM-DD}.parquet
```
