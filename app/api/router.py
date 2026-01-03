from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.api.routes.jobs import router as jobs_router
from app.api.routes.requirements import router as requirements_router
from app.api.routes.tests import router as tests_router
from app.api.routes.reviews import router as reviews_router
from app.api.routes.links import router as links_router
from app.api.routes.settings import router as settings_router
from app.api.routes.public_criteria import router as public_criteria_router
from app.api.routes.coverage import router as coverage_router


settings = get_settings()

api_router = APIRouter(prefix=settings.api_prefix)

# MVP：先保证接口可调用，再逐步补齐 Confluence/XMind 解析与评审计算逻辑
api_router.include_router(jobs_router, tags=["任务"])
api_router.include_router(requirements_router, tags=["需求"])
api_router.include_router(tests_router, tags=["测试"])
api_router.include_router(reviews_router, tags=["评审"])
api_router.include_router(links_router, tags=["链接"])
api_router.include_router(settings_router, tags=["设置"])

# 公共测试用例覆盖度分析
api_router.include_router(public_criteria_router, tags=["公共测试标准"])
api_router.include_router(coverage_router, tags=["覆盖度分析"])

