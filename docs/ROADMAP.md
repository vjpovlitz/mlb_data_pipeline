# MLB Pipeline Roadmap

## Current State (Phases 1-3 complete)

- **Ingestion**: MLB Stats API client, feed parser, live game poller with diff algorithm
- **Storage**: SQL Server (relational), Parquet (date-partitioned), DuckDB (analytical)
- **Streaming**: Redis Streams for real-time event fan-out
- **Processing**: Polars transforms (pitch sequencing, zones, context, counts) + aggregations
- **Orchestration**: 4 Airflow DAGs running in Docker
- **Tests**: 142 passing, ~81% coverage

---

## Phase 4: Win Probability Model

The original end goal. Binary classifier (home team wins) or real-time probability curve per pitch.

- [ ] Feature engineering pipeline — training dataset from enriched pitches + game context (leverage index, base/out state, score diff, pitcher arsenal stats, batter rolling stats)
- [ ] PyTorch model — binary classifier or per-pitch probability curve
- [ ] CUDA training on RTX 5080
- [ ] Model registry — versioned artifacts in `data/models/`, tracked via MLflow or simple JSON manifest
- [ ] Inference endpoint — FastAPI serving real-time predictions during live games

## Phase 5: Real-Time Dashboard

- [ ] FastAPI WebSocket server — subscribe to Redis Streams, push live pitch events + win probability to frontend
- [ ] Grafana dashboards (already in Docker):
  - [ ] Live game scoreboard with win probability curve
  - [ ] Pitcher performance (arsenal breakdown, velocity tracking)
  - [ ] Batter hot/cold zones
  - [ ] Team standings + trends
- [ ] Or React frontend if more control than Grafana is needed

## Phase 6: Advanced Analytics

- [ ] Expected batting average (xBA) — exit velocity + launch angle (pybaseball Statcast data)
- [ ] Pitch movement modeling — horizontal/vertical break analysis per pitch type
- [ ] Pitcher fatigue detection — velocity drop-off over pitch count, rolling window
- [ ] Similarity scores — comparable pitchers/batters using embedding vectors
- [ ] Spray charts — hit location visualization per batter
- [ ] Platoon splits — L/R matchup analysis

## Phase 7: Data Quality & Ops Hardening

- [ ] Great Expectations integration (already in pyproject.toml as optional dep)
- [ ] Alerting — Slack/email on DAG failures or data quality violations
- [ ] Backfill tooling — CLI command to backfill historical seasons (2015-2025)
- [ ] Monitoring — Prometheus metrics for ingestion rates, latency, error counts
- [ ] CI/CD — GitHub Actions for test runs on PR, auto-deploy DAGs

## Phase 8: Historical Data & Season Analysis

- [ ] Bulk historical ingest — pybaseball Statcast data (2015+), retrosheet for older seasons
- [ ] Season-over-season comparisons — player trajectory analysis
- [ ] Trade impact analysis — performance before/after team changes
- [ ] Prospect scouting models — MiLB data if available

---

## Quick Wins

- [ ] Trigger a test DAG run — unpause `daily_backfill`, manually trigger, verify end-to-end
- [ ] Backfill opening week — season started March 27, pull first few days of real data
- [ ] Add Statcast columns — pybaseball `statcast()` for exit velocity, launch angle, expected stats
- [ ] CLI `mlb export` command — expose Parquet export from CLI for ad-hoc use
- [ ] Coverage to 90% — fill gaps in CLI tests, parser edge cases
