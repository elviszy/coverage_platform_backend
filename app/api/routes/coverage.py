"""覆盖度分析 API 路由。

提供覆盖度分析的发起、结果查询、报告导出功能。
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.deps import db_session
from app.models import TestsSource, RequirementsPage, PublicTestCriterion
from app.schemas import (
    CoverageAnalyzeRequest,
    CoverageAnalysisResponse,
    CoverageAnalysisSummary,
    CoverageResultItem,
    CategoryCoverageStats,
    MatchedScenario,
    MatchedRequirement,
    CoverageRunListResponse,
    CoverageRunListItem,
)
from app.services import coverage_analyzer
from app.services.report_generator import generate_coverage_report


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coverage", tags=["覆盖度分析"])


@router.post("/analyze")
async def analyze_coverage(
    payload: CoverageAnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(db_session),
):
    """
    发起一次覆盖度分析任务。
    
    分析将在后台异步执行，返回任务 ID 后可通过 GET /{run_id} 查询结果。
    """
    # 验证 XMind 来源是否存在
    source = db.get(TestsSource, uuid.UUID(payload.xmind_source_id))
    if not source:
        raise HTTPException(status_code=404, detail="XMind 来源不存在")
    
    # 验证需求页面是否存在（如果指定）
    if payload.requirements_page_ids:
        for page_id in payload.requirements_page_ids:
            page = db.get(RequirementsPage, page_id)
            if not page:
                raise HTTPException(status_code=404, detail=f"需求页面 {page_id} 不存在")
    
    # 检查是否有已索引的公共标准
    from app.services.public_criteria_service import get_criteria_with_embeddings
    criteria = get_criteria_with_embeddings(db)
    if not criteria:
        raise HTTPException(
            status_code=400, 
            detail="没有已索引的公共测试标准，请先导入并索引公共标准"
        )
    
    # 构建配置
    config = {}
    if payload.config:
        config = payload.config.model_dump()
    
    # 创建分析任务
    run = coverage_analyzer.create_coverage_run(
        db=db,
        xmind_source_id=payload.xmind_source_id,
        requirements_page_ids=payload.requirements_page_ids,
        config=config,
    )
    
    # 在后台执行分析
    background_tasks.add_task(
        coverage_analyzer.run_coverage_analysis,
        db,
        str(run.run_id),
        payload.xmind_source_id,
        payload.requirements_page_ids,
        config,
    )
    
    return {
        "run_id": str(run.run_id),
        "status": run.status,
    }


@router.get("/runs", response_model=CoverageRunListResponse)
async def list_coverage_runs(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(db_session),
):
    """
    获取覆盖度分析历史列表。
    """
    runs, total = coverage_analyzer.list_coverage_runs(db, limit=limit, offset=offset)
    
    items = []
    for run in runs:
        # 获取 XMind 来源信息
        source = db.get(TestsSource, run.xmind_source_id)
        source_name = source.file_name if source else "未知来源"
        
        summary = run.summary or {}
        
        items.append(CoverageRunListItem(
            run_id=str(run.run_id),
            xmind_source_id=str(run.xmind_source_id),
            xmind_source_name=source_name,
            status=run.status,
            coverage_rate=summary.get("coverage_rate"),
            total_criteria=summary.get("total_criteria"),
            created_at=str(run.created_at),
            finished_at=str(run.finished_at) if run.finished_at else None,
        ))
    
    return CoverageRunListResponse(items=items, total=total)


@router.get("/{run_id}", response_model=CoverageAnalysisResponse)
async def get_coverage_result(
    run_id: str,
    db: Session = Depends(db_session),
):
    """
    获取覆盖度分析结果。
    """
    run = coverage_analyzer.get_coverage_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="分析任务不存在")
    
    # 获取 XMind 来源信息
    source = db.get(TestsSource, run.xmind_source_id)
    source_name = source.file_name if source else "未知来源"
    
    # 获取需求页面信息
    req_pages = []
    if run.requirements_page_ids:
        for page_id in run.requirements_page_ids:
            page = db.get(RequirementsPage, page_id)
            if page:
                req_pages.append({
                    "page_id": page.page_id,
                    "title": page.title,
                    "url": page.page_url,
                })
    
    # 获取结果详情
    results = coverage_analyzer.get_coverage_results(db, run_id)
    
    # 获取公共标准信息
    criteria_map = {}
    for r in results:
        criterion = db.get(PublicTestCriterion, r.criterion_id)
        if criterion:
            criteria_map[r.criterion_id] = criterion
    
    # 构建结果项
    def build_result_item(r) -> CoverageResultItem:
        criterion = criteria_map.get(r.criterion_id)
        return CoverageResultItem(
            criterion_id=r.criterion_id,
            category=criterion.category if criterion else "未知",
            test_point=criterion.test_point if criterion else "未知",
            test_content=criterion.test_content if criterion else None,
            status=r.status,
            best_score=r.best_score,
            matched_keywords=r.matched_keywords or [],
            matched_scenarios=[
                MatchedScenario(
                    scenario_id=s.get("scenario_id", ""),
                    title=s.get("title", ""),
                    path=s.get("path", ""),
                    score=s.get("score", 0),
                    matched_keywords=s.get("matched_keywords", []),
                )
                for s in (r.matched_scenarios or [])
            ],
            matched_requirements=[
                MatchedRequirement(
                    page_id=req.get("page_id", ""),
                    page_title=req.get("page_title", ""),
                    text=req.get("text", ""),
                    score=req.get("score", 0),
                )
                for req in (r.matched_requirements or [])
            ],
            llm_verified=r.llm_verified,
            llm_reason=r.llm_reason,
            llm_suggestion=r.llm_suggestion,
        )
    
    covered_items = [build_result_item(r) for r in results if r.status == "covered"]
    partial_items = [build_result_item(r) for r in results if r.status == "partial"]
    missed_items = [build_result_item(r) for r in results if r.status == "missed"]
    
    # 构建汇总
    summary = None
    if run.summary:
        by_category = [
            CategoryCoverageStats(
                category=cat.get("category", ""),
                total=cat.get("total", 0),
                covered=cat.get("covered", 0),
                partial=cat.get("partial", 0),
                missed=cat.get("missed", 0),
                coverage_rate=cat.get("coverage_rate", 0),
            )
            for cat in run.summary.get("by_category", [])
        ]
        
        summary = CoverageAnalysisSummary(
            total_criteria=run.summary.get("total_criteria", 0),
            covered=run.summary.get("covered", 0),
            partial=run.summary.get("partial", 0),
            missed=run.summary.get("missed", 0),
            coverage_rate=run.summary.get("coverage_rate", 0),
            by_category=by_category,
            requirements_linked=run.summary.get("requirements_linked", 0),
            llm_verified_count=run.summary.get("llm_verified_count", 0),
            llm_suggestion_count=run.summary.get("llm_suggestion_count", 0),
        )
    
    return CoverageAnalysisResponse(
        run_id=str(run.run_id),
        status=run.status,
        xmind_source_id=str(run.xmind_source_id),
        xmind_source_name=source_name,
        requirements_pages=req_pages,
        config=run.config or {},
        summary=summary,
        covered_items=covered_items,
        partial_items=partial_items,
        missed_items=missed_items,
        created_at=str(run.created_at),
        finished_at=str(run.finished_at) if run.finished_at else None,
    )


@router.get("/{run_id}/report")
async def download_coverage_report(
    run_id: str,
    db: Session = Depends(db_session),
):
    """
    导出覆盖度分析报告（Markdown 格式）。
    """
    run = coverage_analyzer.get_coverage_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="分析任务不存在")
    
    if run.status != "completed":
        raise HTTPException(status_code=400, detail="分析任务未完成")
    
    results = coverage_analyzer.get_coverage_results(db, run_id)
    
    # 生成报告
    report_content = generate_coverage_report(db, run, results)
    
    # 获取文件名
    source = db.get(TestsSource, run.xmind_source_id)
    source_name = source.file_name.replace('.xmind', '') if source else 'coverage'
    filename = f"{source_name}_coverage_report.md"
    
    return PlainTextResponse(
        content=report_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

