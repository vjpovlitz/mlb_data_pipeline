"""CLI entry points for the MLB pipeline."""

import asyncio

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


@main.group()
def ingest():
    """Data ingestion commands."""
    pass


@ingest.command("live")
def ingest_live():
    """Start live game poller for real-time data ingestion."""
    click.echo("Starting live game poller... (Phase 2)")
    # Will be implemented in Phase 2


@ingest.command("backfill")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
def ingest_backfill(start: str, end: str):
    """Load historical data for a date range."""
    click.echo(f"Backfilling data from {start} to {end}... (Phase 3)")
    # Will be implemented in Phase 3


@main.command("replay")
@click.argument("game_pk", type=int)
@click.option("--speed", default=10.0, help="Replay speed multiplier")
def replay(game_pk: int, speed: float):
    """Replay a completed game through the pipeline."""
    click.echo(f"Replaying game {game_pk} at {speed}x speed... (Phase 3)")
    # Will be implemented in Phase 3


@main.group()
def db():
    """Database management commands."""
    pass


@db.command("migrate")
def db_migrate():
    """Run database migrations."""
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


@main.command("api-test")
@click.option("--date", default=None, help="Date to query (YYYY-MM-DD)")
def api_test(date: str | None):
    """Test MLB Stats API connectivity."""

    async def _test():
        from datetime import date as dt_date

        from mlb_pipeline.ingestion.client import MLBStatsClient

        if date is None:
            query_date = dt_date.today().isoformat()
        else:
            query_date = date

        async with MLBStatsClient() as client:
            schedule = await client.get_schedule(query_date)
            dates = schedule.get("dates", [])
            if not dates:
                click.echo(f"No games scheduled for {query_date}")
                return
            games = dates[0].get("games", [])
            click.echo(f"Found {len(games)} games for {query_date}:")
            for g in games:
                away = g["teams"]["away"]["team"]["name"]
                home = g["teams"]["home"]["team"]["name"]
                status = g["status"]["detailedState"]
                click.echo(f"  {g['gamePk']}: {away} @ {home} — {status}")

    asyncio.run(_test())


@main.group()
def ml():
    """Machine learning commands."""
    pass


@ml.command("train")
@click.option("--data", required=True, help="Path to training data Parquet")
@click.option("--epochs", default=30, help="Number of training epochs")
def ml_train(data: str, epochs: int):
    """Train win probability model."""
    click.echo(f"Training model on {data} for {epochs} epochs... (Phase 4)")


@ml.command("evaluate")
@click.option("--model", required=True, help="Path to model checkpoint")
def ml_evaluate(model: str):
    """Evaluate model performance."""
    click.echo(f"Evaluating model {model}... (Phase 4)")


@main.command("dashboard")
@click.option("--port", default=8000, help="API server port")
def dashboard(port: int):
    """Start the FastAPI dashboard server."""
    click.echo(f"Starting dashboard API on port {port}... (Phase 6)")


if __name__ == "__main__":
    main()
