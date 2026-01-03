from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from app.api.router import api_router


# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="测试覆盖率评审平台 API",
        version="0.1.0",
        description="用于导入 Confluence 需求（验收标准表格行级）与 XMind 测试场景，分别写入两套向量库（pgvector），运行覆盖度 + 质量评审，并导出评审报告。",
    )

    app.include_router(api_router)
    
    @app.on_event("startup")
    async def startup_warmup():
        """启动时预热：预加载分词库和预热数据库连接"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 预加载 jieba 分词库
        try:
            import jieba
            jieba.initialize()
            logger.info("jieba 分词库已预加载")
        except Exception as e:
            logger.warning(f"jieba 预加载失败: {e}")
        
        # 预热数据库连接
        try:
            from sqlalchemy import text
            from app.db import SessionLocal
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            logger.info("数据库连接池已预热")
        except Exception as e:
            logger.warning(f"数据库预热失败: {e}")
    
    return app


app = create_app()
