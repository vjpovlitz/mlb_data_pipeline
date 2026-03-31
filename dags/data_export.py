"""DAG: Data export — bulk export SQL Server data to Parquet and compact partitions.

Runs weekly on Mondays to export the past week's data and compact monthly partitions.
Also provides a manual trigger for full-season exports.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlb_pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    "data_export",
    default_args=default_args,
    description="Export SQL Server data to Parquet and maintain partitions",
    schedule="0 6 * * 1",  # Every Monday at 6 AM UTC
    start_date=datetime(2026, 3, 27),
    catchup=False,
    tags=["mlb", "export"],
)


def export_weekly_data(**context):
    """Export the past week of game data to Parquet."""
    from datetime import date as dt_date, timedelta as td
    from pathlib import Path

    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings
    from mlb_pipeline.storage.parquet import ParquetStore

    end_date = dt_date.fromisoformat(context["ds"])
    start_date = end_date - td(days=7)

    engine = create_engine(settings.db_connection_string)
    store = ParquetStore(settings.parquet_dir)

    result = store.export_date_range(engine, start_date, end_date)
    print(f"Weekly export {start_date} to {end_date}: {result}")

    # Also export games summary for the season
    season_start = dt_date(end_date.year, 3, 1)
    games_path = store.export_games(engine, season_start, end_date)
    print(f"Season games exported to {games_path}")

    return result


def compact_monthly_partitions(**context):
    """Compact last month's daily Parquet files into monthly files."""
    from datetime import date as dt_date

    from mlb_pipeline.config import settings
    from mlb_pipeline.storage.parquet import ParquetStore

    current_date = dt_date.fromisoformat(context["ds"])

    # Compact previous month
    if current_date.month == 1:
        year, month = current_date.year - 1, 12
    else:
        year, month = current_date.year, current_date.month - 1

    store = ParquetStore(settings.parquet_dir)

    for dataset in ["pitches", "at_bats"]:
        try:
            path = store.compact_date_partitions(dataset, year, month)
            print(f"Compacted {dataset} for {year}-{month:02d} → {path}")
        except FileNotFoundError:
            print(f"No {dataset} data to compact for {year}-{month:02d}")


def export_enriched_pitches(**context):
    """Export pitches with all analytical features added."""
    from datetime import date as dt_date, timedelta as td

    import polars as pl
    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings
    from mlb_pipeline.processing.transforms import enrich_pitches

    end_date = dt_date.fromisoformat(context["ds"])
    start_date = end_date - td(days=7)

    engine = create_engine(settings.db_connection_string)

    with engine.connect() as conn:
        pitches_df = pl.read_database(
            f"SELECT p.* FROM pitches p JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE g.game_date BETWEEN '{start_date}' AND '{end_date}' "
            f"AND g.status = 'Final'",
            connection=conn,
        )

    if pitches_df.is_empty():
        print("No pitches to enrich")
        return

    enriched = enrich_pitches(pitches_df.lazy()).collect()

    output_dir = settings.parquet_dir / "enriched"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pitches_{start_date}_{end_date}.parquet"
    enriched.write_parquet(output_path)

    print(f"Exported {enriched.height} enriched pitches to {output_path}")
    return {"pitches": enriched.height, "path": str(output_path)}


export_task = PythonOperator(
    task_id="export_weekly_data",
    python_callable=export_weekly_data,
    dag=dag,
)

compact_task = PythonOperator(
    task_id="compact_monthly_partitions",
    python_callable=compact_monthly_partitions,
    dag=dag,
)

enrich_task = PythonOperator(
    task_id="export_enriched_pitches",
    python_callable=export_enriched_pitches,
    dag=dag,
)

export_task >> [compact_task, enrich_task]
