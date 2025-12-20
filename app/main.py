from __future__ import annotations

from fastapi import FastAPI

from app.api.router import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="测试覆盖率评审平台 API",
        version="0.1.0",
        description="用于导入 Confluence 需求（验收标准表格行级）与 XMind 测试场景，分别写入两套向量库（pgvector），运行覆盖度 + 质量评审，并导出评审报告。",
    )

    app.include_router(api_router)
    return app


app = create_app()
