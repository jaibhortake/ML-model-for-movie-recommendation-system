"""Configuration management for the movie recommendation system."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="RECSYS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Directories
    model_dir: Path = Field(
        default=Path("models"),
        description="Directory path for trained model artifacts",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory path for datasets",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="Host interface for FastAPI server",
    )
    api_port: int = Field(
        default=8000,
        description="Port for FastAPI server",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Recommendation System Parameters
    model_type: str = Field(
        default="hybrid",
        description="Type of recommendation model (collaborative, content, hybrid)",
    )
    n_factors: int = Field(
        default=50,
        description="Number of latent factors for matrix factorization / collaborative models",
    )
    n_recommendations: int = Field(
        default=10,
        description="Default number of recommendations returned per query",
    )
    collaborative_weight: float = Field(
        default=0.6,
        description="Weight assigned to collaborative filtering in hybrid recommendation",
    )
    content_weight: float = Field(
        default=0.4,
        description="Weight assigned to content-based filtering in hybrid recommendation",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
