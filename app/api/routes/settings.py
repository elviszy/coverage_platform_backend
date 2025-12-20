from __future__ import annotations

from fastapi import APIRouter

from app.schemas import LlmSettingsRequest, OkResponse


router = APIRouter()


@router.put("/settings/llm", response_model=OkResponse)
def update_llm_settings(_: LlmSettingsRequest):
    """更新 LLM 配置（MVP）。

    说明：
    - 生产环境建议使用环境变量或密钥管理服务，不建议提供在线更新接口。
    - 当前版本仅返回 ok。
    """

    return OkResponse(ok=True)
