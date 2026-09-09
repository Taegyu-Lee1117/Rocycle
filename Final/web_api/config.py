"""Environment based configuration for the UI/DB API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


FINAL_DIR = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding another dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    api_host: str
    api_port: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_file(FINAL_DIR / ".env")
        return cls(
            db_host=os.getenv("DB_HOST", "127.0.0.1"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "recycle_db"),
            db_user=os.getenv("DB_USER", "recycle_user"),
            db_password=os.getenv("DB_PASSWORD", ""),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
        )
