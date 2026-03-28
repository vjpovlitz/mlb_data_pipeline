"""Application configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MLB_")

    # SQL Server
    db_server: str = "localhost"
    db_name: str = "mlb_pipeline"
    db_driver: str = "ODBC Driver 17 for SQL Server"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Polling intervals (seconds)
    poll_interval_live: float = 2.0
    poll_interval_idle: float = 60.0
    poll_interval_pregame: float = 30.0

    # Storage paths
    data_dir: Path = Path("data")

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / "parquet"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def db_connection_string(self) -> str:
        return (
            f"mssql+pyodbc://@{self.db_server}/{self.db_name}"
            f"?driver={self.db_driver.replace(' ', '+')}"
            f"&trusted_connection=yes"
            f"&TrustServerCertificate=yes"
        )

    @property
    def db_pyodbc_string(self) -> str:
        return (
            f"DRIVER={{{self.db_driver}}};"
            f"SERVER={self.db_server};"
            f"DATABASE={self.db_name};"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes;"
        )


settings = Settings()
