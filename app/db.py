from __future__ import annotations

import logging
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from app.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)


# 优化的连接池配置，应对不稳定的数据库连接
engine = create_engine(
    settings.database_url,
    # 连接池配置
    poolclass=QueuePool,
    pool_size=5,              # 连接池大小
    max_overflow=10,          # 允许的最大溢出连接数
    pool_timeout=30,          # 获取连接的超时时间（秒）
    pool_recycle=300,         # 连接回收时间（5分钟），防止服务端断开空闲连接
    pool_pre_ping=True,       # 每次使用前检查连接是否有效
    # 连接参数
    connect_args={
        "connect_timeout": 10,           # 连接超时（秒）
        "keepalives": 1,                 # 启用 TCP keepalive
        "keepalives_idle": 30,           # 空闲多久后发送 keepalive
        "keepalives_interval": 10,       # keepalive 间隔
        "keepalives_count": 5,           # keepalive 失败次数后断开
    },
)


# 连接事件监听，用于调试
@event.listens_for(engine, "connect")
def on_connect(dbapi_conn, connection_record):
    logger.debug("数据库连接已建立")


@event.listens_for(engine, "checkout")
def on_checkout(dbapi_conn, connection_record, connection_proxy):
    logger.debug("从连接池获取连接")


@event.listens_for(engine, "checkin")
def on_checkin(dbapi_conn, connection_record):
    logger.debug("连接已归还连接池")


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：获取数据库会话。"""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """测试数据库连接是否正常。"""
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"数据库连接测试失败: {e}")
        return False

