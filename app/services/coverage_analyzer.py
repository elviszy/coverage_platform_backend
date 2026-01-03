"""覆盖度分析核心引擎。

提供 XMind 测试点与公共测试标准的匹配分析功能，支持：
- 三方关联（XMind ↔ 需求文档 ↔ 公共标准）
- 相似度 + 关键词双重校验
- 多级阈值动态调整
- 按类型分组统计
- LLM 增强（边界验证 + 智能建议）
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    PublicTestCriterion,
    CoverageAnalysisRun,
    CoverageAnalysisResult,
    TestScenario,
    TestsSource,
    RequirementsPage,
    RequirementCriterion,
)
from app.services.embedding import embed_text, cosine_similarity
from app.services.public_criteria_service import get_criteria_with_embeddings


logger = logging.getLogger(__name__)


# ==================== 多级阈值调整 ====================


def get_dynamic_threshold(
    text_length: int,
    base_cover: float = 0.80,
    base_partial: float = 0.65,
) -> Tuple[float, float]:
    """
    根据文本长度动态调整匹配阈值。
    
    说明：
    - 短文本信息量不足，降低阈值避免误判
    - 长文本信息量充足，使用标准阈值
    
    Args:
        text_length: 测试点文本长度
        base_cover: 基础覆盖阈值
        base_partial: 基础部分覆盖阈值
    
    Returns:
        (cover_threshold, partial_threshold)
    """
    if text_length < 10:  # 极短描述（如"分页"）
        return (base_cover - 0.10, base_partial - 0.10)  # (0.70, 0.55)
    elif text_length < 20:  # 短描述（如"删除后刷新"）
        return (base_cover - 0.05, base_partial - 0.05)  # (0.75, 0.60)
    elif text_length < 50:  # 中等描述
        return (base_cover, base_partial)  # (0.80, 0.65)
    else:  # 详细描述
        return (base_cover + 0.02, base_partial + 0.02)  # (0.82, 0.67)


# ==================== 关键词匹配 ====================


def calculate_keyword_score(
    scenario_text: str,
    keywords: List[str],
) -> Tuple[float, List[str]]:
    """
    计算关键词命中分数。
    
    Args:
        scenario_text: 测试场景文本
        keywords: 公共标准的关键词列表
        
    Returns:
        (score, matched_keywords)
        score 范围 0.0 ~ 1.0
    """
    if not keywords:
        return 0.0, []
    
    matched = [kw for kw in keywords if kw in scenario_text]
    score = len(matched) / len(keywords)
    
    return score, matched


def calculate_combined_score(
    embedding_score: float,
    keyword_score: float,
    embedding_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> float:
    """
    计算综合分数（Embedding + 关键词）。
    
    默认权重：
    - Embedding 相似度: 70%
    - 关键词命中: 30%
    """
    return embedding_score * embedding_weight + keyword_score * keyword_weight


# ==================== 按类型分组统计 ====================


def calculate_category_coverage(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按公共标准类型分组统计覆盖率。
    
    Args:
        results: 分析结果列表，每项包含 {category, status, ...}
        
    Returns:
        [
            {"category": "增删改", "total": 25, "covered": 18, "partial": 3, "missed": 4, "coverage_rate": 72.0},
            {"category": "审核", "total": 12, "covered": 10, "partial": 1, "missed": 1, "coverage_rate": 83.3},
            ...
        ]
    """
    stats = defaultdict(lambda: {"total": 0, "covered": 0, "partial": 0, "missed": 0})
    
    for r in results:
        cat = r.get("category", "其他")
        stats[cat]["total"] += 1
        status = r.get("status", "missed")
        if status == "covered":
            stats[cat]["covered"] += 1
        elif status == "partial":
            stats[cat]["partial"] += 1
        else:
            stats[cat]["missed"] += 1
    
    output = []
    for cat, s in stats.items():
        coverage_rate = (s["covered"] / s["total"] * 100) if s["total"] > 0 else 0
        output.append({
            "category": cat,
            "total": s["total"],
            "covered": s["covered"],
            "partial": s["partial"],
            "missed": s["missed"],
            "coverage_rate": round(coverage_rate, 1),
        })
    
    return sorted(output, key=lambda x: x["total"], reverse=True)


# ==================== 场景匹配 ====================


def find_best_matching_scenarios(
    criterion: PublicTestCriterion,
    scenarios: List[TestScenario],
    config: Dict[str, Any],
) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    """
    为公共标准找到最佳匹配的测试场景。
    
    Args:
        criterion: 公共测试标准
        scenarios: 测试场景列表
        config: 配置（embedding_weight, keyword_weight, 等）
        
    Returns:
        (best_score, matched_keywords, matched_scenarios)
    """
    embedding_weight = config.get("embedding_weight", 0.7)
    keyword_weight = config.get("keyword_weight", 0.3)
    
    if criterion.embedding is None:
        logger.warning(f"公共标准 {criterion.criterion_id} 没有 embedding，跳过")
        return 0.0, [], []
    
    # 将 criterion embedding 转换为 numpy array
    criterion_emb = np.array(criterion.embedding)
    
    all_matches = []
    best_score = 0.0
    all_matched_keywords = set()
    
    for scenario in scenarios:
        if scenario.embedding is None:
            continue
        
        # 计算 embedding 相似度
        scenario_emb = np.array(scenario.embedding)
        emb_score = cosine_similarity(criterion_emb, scenario_emb)
        
        # 计算关键词匹配
        scenario_text = f"{scenario.title} {scenario.path} {scenario.notes or ''}"
        kw_score, matched_kw = calculate_keyword_score(scenario_text, criterion.keywords)
        
        # 计算综合分数
        combined_score = calculate_combined_score(emb_score, kw_score, embedding_weight, keyword_weight)
        
        if combined_score > 0.4:  # 仅保留有一定相关性的匹配
            all_matches.append({
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "path": scenario.path,
                "score": round(combined_score, 4),
                "embedding_score": round(emb_score, 4),
                "keyword_score": round(kw_score, 4),
                "matched_keywords": matched_kw,
            })
            all_matched_keywords.update(matched_kw)
            
            if combined_score > best_score:
                best_score = combined_score
    
    # 按分数排序，取 top 5
    all_matches.sort(key=lambda x: x["score"], reverse=True)
    top_matches = all_matches[:5]
    
    return best_score, list(all_matched_keywords), top_matches


# ==================== 需求关联 ====================


def find_related_requirements(
    criterion: PublicTestCriterion,
    requirements: List[RequirementCriterion],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    查找与公共标准相关的需求点。
    
    Args:
        criterion: 公共测试标准
        requirements: 需求标准列表
        top_k: 返回最相关的 k 个
        
    Returns:
        [{page_id, page_title, text, score}, ...]
    """
    if criterion.embedding is None:
        return []
    
    criterion_emb = np.array(criterion.embedding)
    matches = []
    
    for req in requirements:
        if req.embedding is None:
            continue
        
        req_emb = np.array(req.embedding)
        score = cosine_similarity(criterion_emb, req_emb)
        
        if score > 0.5:  # 仅保留有一定相关性的
            matches.append({
                "page_id": req.page_id,
                "page_title": req.title,
                "criterion_id": req.criterion_id,
                "text": req.normalized_text[:200],  # 截断过长文本
                "score": round(score, 4),
            })
    
    # 按分数排序，取 top k
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:top_k]


# ==================== 主流程 ====================


def run_coverage_analysis(
    db: Session,
    run_id: str,
    xmind_source_id: str,
    requirements_page_ids: Optional[List[str]],
    config: Dict[str, Any],
) -> None:
    """
    覆盖度分析主流程。
    
    流程：
    1. 获取 XMind 场景列表
    2. 获取公共测试标准列表
    3. 获取关联的需求文档（可选）
    4. 对每个公共标准：
       a. 计算与 XMind 场景的 Embedding 相似度
       b. 计算关键词命中分数
       c. 综合得分 = Embedding*0.7 + 关键词*0.3
       d. 应用多级阈值（根据文本长度动态调整）
       e. 计算与需求文档的相似度（上下文增强）
       f. 根据阈值判定覆盖状态
    5. 边界项触发 LLM 验证（结合需求上下文）
    6. 未覆盖项生成 LLM 建议（参考需求描述）
    7. 按类型分组统计覆盖率
    8. 保存结果
    """
    logger.info(f"开始覆盖度分析: run_id={run_id}, xmind_source_id={xmind_source_id}")
    
    # 更新状态为运行中
    run = db.get(CoverageAnalysisRun, uuid.UUID(run_id))
    if not run:
        logger.error(f"找不到分析任务: {run_id}")
        return
    
    run.status = "running"
    db.commit()
    
    try:
        # 1. 获取 XMind 场景列表
        scenarios = list(db.execute(
            select(TestScenario).where(TestScenario.source_id == uuid.UUID(xmind_source_id))
        ).scalars().all())
        
        if not scenarios:
            raise ValueError(f"XMind 来源 {xmind_source_id} 没有测试场景")
        
        logger.info(f"获取到 {len(scenarios)} 个测试场景")
        
        # 2. 获取公共测试标准
        categories = config.get("categories")
        criteria = get_criteria_with_embeddings(
            db, 
            category=categories[0] if categories and len(categories) == 1 else None
        )
        
        if categories and len(categories) > 1:
            criteria = [c for c in criteria if c.category in categories]
        
        if not criteria:
            raise ValueError("没有已索引的公共测试标准")
        
        logger.info(f"获取到 {len(criteria)} 个公共测试标准")
        
        # 3. 获取关联的需求文档（可选）
        requirements = []
        if requirements_page_ids:
            requirements = list(db.execute(
                select(RequirementCriterion).where(
                    RequirementCriterion.page_id.in_(requirements_page_ids),
                    RequirementCriterion.is_active == True,
                )
            ).scalars().all())
            logger.info(f"获取到 {len(requirements)} 个需求点")
        
        # 4. 分析每个公共标准
        threshold_cover = config.get("threshold_cover", 0.80)
        threshold_partial = config.get("threshold_partial", 0.65)
        enable_dynamic = config.get("enable_dynamic_threshold", True)
        
        all_results = []
        covered_count = 0
        partial_count = 0
        missed_count = 0
        requirements_linked_count = 0
        
        for criterion in criteria:
            # 计算最佳匹配
            best_score, matched_kw, matched_scenarios = find_best_matching_scenarios(
                criterion, scenarios, config
            )
            
            # 查找相关需求
            matched_reqs = find_related_requirements(criterion, requirements) if requirements else []
            if matched_reqs:
                requirements_linked_count += 1
            
            # 应用多级阈值
            if enable_dynamic:
                text_len = len(criterion.normalized_text)
                cover_th, partial_th = get_dynamic_threshold(text_len, threshold_cover, threshold_partial)
            else:
                cover_th, partial_th = threshold_cover, threshold_partial
            
            # 判定覆盖状态
            if best_score >= cover_th:
                status = "covered"
                covered_count += 1
            elif best_score >= partial_th:
                status = "partial"
                partial_count += 1
            else:
                status = "missed"
                missed_count += 1
            
            # 保存结果
            result = CoverageAnalysisResult(
                run_id=uuid.UUID(run_id),
                criterion_id=criterion.criterion_id,
                status=status,
                best_score=best_score,
                matched_keywords=matched_kw,
                matched_scenarios=matched_scenarios,
                matched_requirements=matched_reqs,
                llm_verified=False,
                llm_reason=None,
                llm_suggestion=None,
            )
            db.add(result)
            
            all_results.append({
                "criterion_id": criterion.criterion_id,
                "category": criterion.category,
                "status": status,
                "best_score": best_score,
            })
        
        # 5. TODO: 边界项 LLM 验证（需要配置 LLM）
        llm_config = config.get("llm")
        llm_verified_count = 0
        llm_suggestion_count = 0
        
        if llm_config and llm_config.get("enable_boundary_verify"):
            # 暂时跳过，后续实现
            logger.info("LLM 边界验证功能待实现")
        
        if llm_config and llm_config.get("enable_miss_suggestion"):
            # 暂时跳过，后续实现
            logger.info("LLM 智能建议功能待实现")
        
        # 6. 按类型分组统计
        by_category = calculate_category_coverage(all_results)
        
        # 7. 保存汇总
        total = len(criteria)
        coverage_rate = (covered_count / total * 100) if total > 0 else 0
        
        run.summary = {
            "total_criteria": total,
            "covered": covered_count,
            "partial": partial_count,
            "missed": missed_count,
            "coverage_rate": round(coverage_rate, 1),
            "by_category": by_category,
            "requirements_linked": requirements_linked_count,
            "llm_verified_count": llm_verified_count,
            "llm_suggestion_count": llm_suggestion_count,
        }
        run.status = "completed"
        run.finished_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"覆盖度分析完成: run_id={run_id}, coverage_rate={coverage_rate:.1f}%")
        
    except Exception as e:
        logger.exception(f"覆盖度分析失败: {e}")
        run.status = "failed"
        run.summary = {"error": str(e)}
        run.finished_at = datetime.utcnow()
        db.commit()
        raise


def create_coverage_run(
    db: Session,
    xmind_source_id: str,
    requirements_page_ids: Optional[List[str]],
    config: Dict[str, Any],
) -> CoverageAnalysisRun:
    """
    创建覆盖度分析任务。
    
    Returns:
        新创建的分析任务
    """
    run = CoverageAnalysisRun(
        xmind_source_id=uuid.UUID(xmind_source_id),
        requirements_page_ids=requirements_page_ids or [],
        status="pending",
        config=config,
        summary={},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    
    return run


def get_coverage_run(db: Session, run_id: str) -> Optional[CoverageAnalysisRun]:
    """获取分析任务。"""
    return db.get(CoverageAnalysisRun, uuid.UUID(run_id))


def get_coverage_results(
    db: Session,
    run_id: str,
) -> List[CoverageAnalysisResult]:
    """获取分析结果详情。"""
    return list(db.execute(
        select(CoverageAnalysisResult).where(
            CoverageAnalysisResult.run_id == uuid.UUID(run_id)
        )
    ).scalars().all())


def list_coverage_runs(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[CoverageAnalysisRun], int]:
    """获取分析历史列表。"""
    # 获取总数
    from sqlalchemy import func
    total = db.execute(
        select(func.count()).select_from(CoverageAnalysisRun)
    ).scalar() or 0
    
    # 获取列表
    runs = list(db.execute(
        select(CoverageAnalysisRun)
        .order_by(CoverageAnalysisRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).scalars().all())
    
    return runs, total
