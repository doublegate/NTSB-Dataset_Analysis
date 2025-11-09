"""
API Configuration Module

Loads configuration from environment variables using pydantic-settings.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(
        default="postgresql://parobek@localhost/ntsb_aviation",
        description="PostgreSQL database connection URL",
    )

    # API
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Auto-reload on code changes")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        description="Comma-separated list of allowed CORS origins",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Pagination
    default_page_size: int = Field(default=100, description="Default page size")
    max_page_size: int = Field(default=1000, description="Maximum page size")

    # Rate Limiting (future use)
    rate_limit_per_minute: int = Field(
        default=100, description="API rate limit per minute"
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# Global settings instance
settings = Settings()
