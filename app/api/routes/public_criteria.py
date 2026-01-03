"""公共测试标准管理 API 路由。

提供公共测试标准的导入、查询、编辑、删除和索引重建功能。
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.deps import db_session
from app.schemas import (
    PublicCriterion,
    PublicCriterionCreate,
    PublicCriterionUpdate,
    PublicCriteriaListResponse,
    PublicCriteriaImportRequest,
    PublicCriteriaImportResponse,
    PublicCriteriaIndexResponse,
)
from app.services import public_criteria_service


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/public-criteria", tags=["公共测试标准"])


@router.post("/import", response_model=PublicCriteriaImportResponse)
async def import_criteria(
    payload: PublicCriteriaImportRequest,
    use_llm: bool = True,
    db: Session = Depends(db_session),
):
    """
    从 Markdown 内容导入公共测试标准。
    
    支持表格格式：
    ```
    类型	测试点	测试内容
    选择客户/商品	正常情况	选择客户落到主界面后...
    ```
    
    Args:
        use_llm: 是否使用 LLM 提取关键词（默认 True）
    """
    result = public_criteria_service.import_criteria(
        db=db,
        content=payload.content,
        replace_all=payload.replace_all,
        use_llm=use_llm,
    )
    
    return PublicCriteriaImportResponse(
        imported=result["imported"],
        updated=result["updated"],
        skipped=result["skipped"],
        errors=result["errors"],
    )


@router.post("/import-file", response_model=PublicCriteriaImportResponse)
async def import_criteria_from_file(
    file: UploadFile = File(...),
    replace_all: bool = False,
    use_llm: bool = True,
    db: Session = Depends(db_session),
):
    """
    从上传的 Markdown 文件导入公共测试标准。
    
    Args:
        use_llm: 是否使用 LLM 提取关键词（默认 True）
    """
    if not file.filename.endswith(('.md', '.txt')):
        raise HTTPException(status_code=400, detail="仅支持 .md 或 .txt 文件")
    
    content = await file.read()
    content_str = content.decode('utf-8')
    
    result = public_criteria_service.import_criteria(
        db=db,
        content=content_str,
        replace_all=replace_all,
        use_llm=use_llm,
    )
    
    return PublicCriteriaImportResponse(
        imported=result["imported"],
        updated=result["updated"],
        skipped=result["skipped"],
        errors=result["errors"],
    )


@router.get("/", response_model=PublicCriteriaListResponse)
async def list_criteria(
    category: Optional[str] = None,
    is_active: Optional[bool] = True,
    search: Optional[str] = None,
    limit: int = 500,
    offset: int = 0,
    db: Session = Depends(db_session),
):
    """
    获取公共测试标准列表。
    
    支持按类型筛选、搜索和分页。
    """
    items, total = public_criteria_service.list_criteria(
        db=db,
        category=category,
        is_active=is_active,
        search=search,
        limit=limit,
        offset=offset,
    )
    
    return PublicCriteriaListResponse(
        items=[
            PublicCriterion(
                criterion_id=c.criterion_id,
                category=c.category,
                test_point=c.test_point,
                test_content=c.test_content,
                keywords=c.keywords or [],
                is_active=c.is_active,
                created_at=str(c.created_at),
                updated_at=str(c.updated_at),
            )
            for c in items
        ],
        total=total,
    )


@router.get("/categories")
async def get_categories(
    db: Session = Depends(db_session),
):
    """
    获取所有测试类型列表。
    """
    categories = public_criteria_service.get_categories(db)
    return {"categories": categories}


@router.get("/{criterion_id}", response_model=PublicCriterion)
async def get_criterion(
    criterion_id: str,
    db: Session = Depends(db_session),
):
    """
    根据 ID 获取单个公共测试标准。
    """
    criterion = public_criteria_service.get_criterion(db, criterion_id)
    if not criterion:
        raise HTTPException(status_code=404, detail="测试标准不存在")
    
    return PublicCriterion(
        criterion_id=criterion.criterion_id,
        category=criterion.category,
        test_point=criterion.test_point,
        test_content=criterion.test_content,
        keywords=criterion.keywords or [],
        is_active=criterion.is_active,
        created_at=str(criterion.created_at),
        updated_at=str(criterion.updated_at),
    )


@router.put("/{criterion_id}", response_model=PublicCriterion)
async def update_criterion(
    criterion_id: str,
    payload: PublicCriterionUpdate,
    db: Session = Depends(db_session),
):
    """
    更新公共测试标准。
    """
    criterion = public_criteria_service.update_criterion(
        db=db,
        criterion_id=criterion_id,
        data=payload.model_dump(exclude_unset=True),
    )
    
    if not criterion:
        raise HTTPException(status_code=404, detail="测试标准不存在")
    
    return PublicCriterion(
        criterion_id=criterion.criterion_id,
        category=criterion.category,
        test_point=criterion.test_point,
        test_content=criterion.test_content,
        keywords=criterion.keywords or [],
        is_active=criterion.is_active,
        created_at=str(criterion.created_at),
        updated_at=str(criterion.updated_at),
    )


@router.delete("/{criterion_id}")
async def delete_criterion(
    criterion_id: str,
    db: Session = Depends(db_session),
):
    """
    删除公共测试标准。
    """
    success = public_criteria_service.delete_criterion(db, criterion_id)
    if not success:
        raise HTTPException(status_code=404, detail="测试标准不存在")
    
    return {"ok": True}


@router.post("/index", response_model=PublicCriteriaIndexResponse)
async def index_criteria(
    force: bool = False,
    db: Session = Depends(db_session),
):
    """
    重建公共测试标准的 Embedding 索引。
    
    Args:
        force: 是否强制重建所有索引（默认仅索引未生成 embedding 的记录）
    """
    result = public_criteria_service.index_criteria(db, force=force)
    
    return PublicCriteriaIndexResponse(
        indexed=result["indexed"],
        failed=result["failed"],
    )
