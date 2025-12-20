from __future__ import annotations

from sqlalchemy.orm import Session

from fastapi import Depends

from app.db import get_db


def db_session(db: Session = Depends(get_db)) -> Session:
    """依赖注入：数据库会话。"""

    return db
