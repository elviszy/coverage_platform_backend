from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


settings = get_settings()


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：获取数据库会话。"""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
