"""CLI entry points for the MLB pipeline."""

import asyncio
from datetime import date

import click
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger()


@click.group()
def main():
    """MLB Real-Time Data Pipeline CLI."""
    pass


# ====================================================================
# INGEST
# ====================================================================

@main.group()
def ingest():
    """Data ingestion commands."""
    pass


@ingest.command("live")
@click.option("--date", "target_date", default=None, help="Date to poll (YYYY-MM-DD, default: today)")
def ingest_live(target_date: str | None):
    """Start live game poller for real-time data ingestion."""
    from mlb_pipeline.ingestion.poller import LiveGamePoller

    d = date.fromisoformat(target_date) if target_date else date.today()
    click.echo(f"Starting live game poller for {d}...")
    asyncio.run(LiveGamePoller(d).run())


@ingest.command("backfill")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--no-sql", is_flag=True, default=False, help="Skip SQL Server writes")
@click.option("--no-parquet", is_flag=True, default=False, help="Skip Parquet writes")
@click.option("--concurrency", default=4, help="Max concurrent game fetches")
def ingest_backfill(start: str, end: str, no_sql: bool, no_parquet: bool, concurrency: int):
    """Load historical data for a date range."""
    from mlb_pipeline.storage.historical_loader import HistoricalLoader

    async def _run():
        loader = HistoricalLoader()
        result = await loader.load_date_range(
            date.fromisoformat(start),
            date.fromisoformat(end),
            write_sql=not no_sql,
            write_parquet=not no_parquet,
            max_concurrent=concurrency,
        )
        click.echo(
            f"Done: {result['success']}/{result['total']} games loaded"
            f" ({result['failed']} failed)"
        )

    asyncio.run(_run())


# ====================================================================
# REPLAY
# ====================================================================

@main.command("replay")
@click.argument("game_pk", type=int)
@click.option("--speed", default=10.0, help="Replay speed multiplier (default: 10x)")
def replay(game_pk: int, speed: float):
    """Replay a completed game through Redis Streams for pipeline testing."""
    from mlb_pipeline.storage.historical_loader import HistoricalLoader

    async def _run():
        loader = HistoricalLoader()
        click.echo(f"Replaying game {game_pk} at {speed}x speed...")
        await loader.replay_game(game_pk, speed=speed)
        click.echo("Replay complete.")

    asyncio.run(_run())


# ====================================================================
# DB
# ====================================================================

@main.group()
def db():
    """Database management commands."""
    pass


@db.command("migrate")
def db_migrate():
    """Run Alembic database migrations."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    click.echo("Migrations applied successfully.")


@db.command("seed-teams")
def db_seed_teams():
    """Seed teams and players reference data from MLB API."""

    async def _seed():
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.client import MLBStatsClient
        from mlb_pipeline.models.database import Base, Team

        engine = create_engine(settings.db_connection_string)
        async with MLBStatsClient() as client:
            data = await client.get_teams(season=2025)
            teams = data.get("teams", [])

        with Session(engine) as session:
            for t in teams:
                team = Team(
                    team_id=t["id"],
                    name=t["name"],
                    abbreviation=t.get("abbreviation", ""),
                    league=t.get("league", {}).get("name", ""),
                    division=t.get("division", {}).get("name", ""),
                    venue_name=t.get("venue", {}).get("name"),
                )
                session.merge(team)
            session.commit()
            click.echo(f"Seeded {len(teams)} teams.")

    asyncio.run(_seed())


@db.command("test-connection")
def db_test_connection():
    """Test SQL Server connectivity."""
    from sqlalchemy import create_engine, text

    from mlb_pipeline.config import settings

    engine = create_engine(settings.db_connection_string)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT @@VERSION")).scalar()
        click.echo(f"Connected to: {result[:80]}...")


# ====================================================================
# ML
# ====================================================================

@main.group()
def ml():
    """Machine learning commands."""
    pass


@ml.command("prepare-data")
@click.option("--season", default=None, type=int, help="Season year (default: current year)")
@click.option("--output", default=None, help="Output Parquet path")
def ml_prepare_data(season: int | None, output: str | None):
    """Build win-probability training dataset from Parquet store."""
    import polars as pl

    from mlb_pipeline.config import settings
    from mlb_pipeline.processing.feature_engineer import build_training_dataset
    from mlb_pipeline.storage.parquet_store import ParquetStore

    s = season or date.today().year
    store = ParquetStore()
    click.echo(f"Loading at-bats for season {s}...")
    at_bats_df = store.load_at_bats(season=s)

    click.echo("Loading game outcomes...")
    games_df = store.query(
        "SELECT game_pk, home_team_won FROM games WHERE home_team_won IS NOT NULL"
        if False  # DuckDB doesn't have games table — use parquet path
        else "SELECT game_pk, 1 AS home_team_won FROM at_bats LIMIT 0"
    )

    # If SQL Server is available, load game outcomes from there
    try:
        from sqlalchemy import create_engine, text as sqla_text
        engine = create_engine(settings.db_connection_string)
        with engine.connect() as conn:
            rows = conn.execute(sqla_text(
                "SELECT game_pk, home_team_won FROM games WHERE home_team_won IS NOT NULL"
            )).fetchall()
        games_df = pl.DataFrame([{"game_pk": r[0], "home_team_won": bool(r[1])} for r in rows])
        click.echo(f"Loaded {len(games_df)} game outcomes from SQL Server.")
    except Exception as exc:
        click.echo(f"Warning: Could not load game outcomes from DB ({exc}). Using dummy labels.")
        game_pks = at_bats_df["game_pk"].unique().to_list()
        games_df = pl.DataFrame([
            {"game_pk": gp, "home_team_won": (i % 2 == 0)}
            for i, gp in enumerate(game_pks)
        ])

    click.echo("Engineering features...")
    training_df = build_training_dataset(at_bats_df, games_df)

    out_path = output or str(settings.parquet_dir / "win_prob_training" / f"training_{s}.parquet")
    from mlb_pipeline.storage.parquet_store import ParquetStore as PS
    ps = PS()
    ps.write_training_dataset(training_df, f"training_{s}")
    click.echo(f"Dataset written: {len(training_df)} rows → {out_path}")


@ml.command("train")
@click.option("--data", required=True, help="Path to training Parquet dataset")
@click.option("--model-type", default="mlp", type=click.Choice(["mlp", "lstm"]), help="Model architecture")
@click.option("--epochs", default=50, help="Max training epochs")
@click.option("--batch-size", default=512, help="Batch size")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--hidden-dim", default=256, help="Hidden dimension (MLP)")
@click.option("--output-dir", default=None, help="Output directory for checkpoints")
def ml_train(data: str, model_type: str, epochs: int, batch_size: int, lr: float, hidden_dim: int, output_dir: str | None):
    """Train win probability model (MLP or LSTM)."""
    from mlb_pipeline.config import settings
    from mlb_pipeline.ml.trainer import ModelTrainer, TrainingConfig

    cfg = TrainingConfig(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        hidden_dim=hidden_dim,
    )
    trainer = ModelTrainer(cfg)
    out = output_dir or str(settings.model_dir)

    click.echo(f"Training {model_type.upper()} on {data} for up to {epochs} epochs...")
    if model_type == "mlp":
        result = trainer.train_mlp(data, out)
    else:
        result = trainer.train_lstm(data, out)

    click.echo(f"\nTraining complete!")
    click.echo(f"  Best epoch:   {result.best_epoch}/{result.total_epochs}")
    click.echo(f"  Best val AUC: {result.best_val_auc:.4f}")
    click.echo(f"  Model saved:  {result.model_path}")
    click.echo(f"  Metrics:      {result.metrics_path}")


@ml.command("evaluate")
@click.option("--model", required=True, help="Path to model checkpoint (.pt)")
@click.option("--data", required=True, help="Path to evaluation Parquet dataset")
@click.option("--model-type", default="mlp", type=click.Choice(["mlp", "lstm"]))
def ml_evaluate(model: str, data: str, model_type: str):
    """Evaluate model performance with AUC, Brier score, and calibration."""
    from mlb_pipeline.ml.trainer import ModelEvaluator

    click.echo(f"Evaluating {model} on {data}...")
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(data, model_type=model_type)

    click.echo(f"\nEvaluation Results:")
    click.echo(f"  AUC:          {results['auc']:.4f}")
    click.echo(f"  Brier Score:  {results['brier_score']:.4f}")
    click.echo(f"  Log Loss:     {results['log_loss']:.4f}")
    click.echo(f"  Test Samples: {results['n_test_samples']}")
    click.echo(f"  Test Games:   {results['n_test_games']}")


@ml.command("predict")
@click.option("--model", default=None, help="Model checkpoint path (optional)")
@click.option("--inning", default=7, type=int)
@click.option("--half", default="top", type=click.Choice(["top", "bottom"]))
@click.option("--outs", default=0, type=int)
@click.option("--away-score", default=2, type=int)
@click.option("--home-score", default=3, type=int)
@click.option("--runner-1b", is_flag=True)
@click.option("--runner-2b", is_flag=True)
@click.option("--runner-3b", is_flag=True)
def ml_predict(
    model, inning, half, outs, away_score, home_score,
    runner_1b, runner_2b, runner_3b
):
    """Run a single win probability prediction from command-line game state."""
    from mlb_pipeline.ml.inference import WinProbInferenceService

    svc = WinProbInferenceService(model)
    if model:
        svc.load(model)
    else:
        svc.load_fallback()

    prob = svc.predict_state(
        inning=inning,
        half_inning=half,
        outs=outs,
        away_score=away_score,
        home_score=home_score,
        runner_on_first=runner_1b,
        runner_on_second=runner_2b,
        runner_on_third=runner_3b,
    )
    click.echo(f"\nGame State: Inn {half[0].upper()}{inning}, {outs} outs, Away {away_score} — Home {home_score}")
    click.echo(f"Home Win Probability:  {prob:.1%}")
    click.echo(f"Away Win Probability:  {1-prob:.1%}")
    click.echo(f"Model: {svc.model_version}")


# ====================================================================
# API
# ====================================================================

@main.command("api-test")
@click.option("--date", "query_date", default=None, help="Date to query (YYYY-MM-DD)")
def api_test(query_date: str | None):
    """Test MLB Stats API connectivity."""

    async def _test():
        from mlb_pipeline.ingestion.client import MLBStatsClient

        d = query_date or str(date.today())
        async with MLBStatsClient() as client:
            schedule = await client.get_schedule(d)
            dates = schedule.get("dates", [])
            if not dates:
                click.echo(f"No games scheduled for {d}")
                return
            games = dates[0].get("games", [])
            click.echo(f"Found {len(games)} games for {d}:")
            for g in games:
                away = g["teams"]["away"]["team"]["name"]
                home = g["teams"]["home"]["team"]["name"]
                status = g["status"]["detailedState"]
                click.echo(f"  {g['gamePk']}: {away} @ {home} — {status}")

    asyncio.run(_test())


# ====================================================================
# DASHBOARD
# ====================================================================

@main.command("dashboard")
@click.option("--port", default=8000, help="API server port")
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--reload", is_flag=True, default=False, help="Hot reload (dev mode)")
@click.option("--model", default=None, help="Model checkpoint to pre-load")
def dashboard(port: int, host: str, reload: bool, model: str | None):
    """Start the FastAPI dashboard server with WebSocket support."""
    import uvicorn

    if model:
        import os
        os.environ["MLB_MODEL_PATH"] = model

    click.echo(f"Starting MLB dashboard on http://{host}:{port}")
    click.echo("  Frontend:  http://localhost:{port}/")
    click.echo("  API docs:  http://localhost:{port}/docs")
    uvicorn.run(
        "mlb_pipeline.dashboard.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
