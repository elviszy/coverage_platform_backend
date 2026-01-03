"""Markdown 报告生成服务。

生成覆盖度分析的 Markdown 格式报告，包含：
- 概览信息
- 汇总统计
- 按类型分组统计
- 未覆盖项及 AI 建议
- 已覆盖项列表
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from app.models import (
    CoverageAnalysisRun,
    CoverageAnalysisResult,
    PublicTestCriterion,
    TestsSource,
    RequirementsPage,
)


logger = logging.getLogger(__name__)


def generate_coverage_report(
    db: Session,
    run: CoverageAnalysisRun,
    results: List[CoverageAnalysisResult],
) -> str:
    """
    生成覆盖度分析 Markdown 报告。
    
    Args:
        db: 数据库会话
        run: 分析任务
        results: 分析结果列表
        
    Returns:
        Markdown 格式的报告内容
    """
    # 获取 XMind 来源信息
    source = db.get(TestsSource, run.xmind_source_id)
    source_name = source.file_name if source else "未知来源"
    
    # 获取需求页面信息
    req_pages = []
    if run.requirements_page_ids:
        for page_id in run.requirements_page_ids:
            page = db.get(RequirementsPage, page_id)
            if page:
                req_pages.append({"title": page.title, "url": page.page_url})
    
    # 获取公共标准的详细信息
    criteria_map = {}
    for r in results:
        criterion = db.get(PublicTestCriterion, r.criterion_id)
        if criterion:
            criteria_map[r.criterion_id] = criterion
    
    # 分类结果
    covered_items = [r for r in results if r.status == "covered"]
    partial_items = [r for r in results if r.status == "partial"]
    missed_items = [r for r in results if r.status == "missed"]
    
    summary = run.summary or {}
    
    # 构建报告
    lines = []
    
    # 标题
    lines.append("# 公共测试用例覆盖度分析报告")
    lines.append("")
    
    # 概览
    lines.append("## 概览")
    lines.append("")
    lines.append(f"- **分析时间**: {run.created_at.strftime('%Y-%m-%d %H:%M') if isinstance(run.created_at, datetime) else run.created_at}")
    lines.append(f"- **用例来源**: {source_name}")
    
    if req_pages:
        req_links = ", ".join([f"[{p['title']}]({p['url']})" for p in req_pages])
        lines.append(f"- **关联需求**: {req_links}")
    
    coverage_rate = summary.get("coverage_rate", 0)
    lines.append(f"- **覆盖率**: **{coverage_rate:.1f}%**")
    
    llm_verified = summary.get("llm_verified_count", 0)
    llm_suggestions = summary.get("llm_suggestion_count", 0)
    if llm_verified > 0:
        lines.append(f"- **LLM 验证**: {llm_verified} 项边界匹配")
    if llm_suggestions > 0:
        lines.append(f"- **智能建议**: {llm_suggestions} 项未覆盖")
    
    lines.append("")
    
    # 汇总
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 状态 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    
    total = summary.get("total_criteria", len(results))
    covered = summary.get("covered", len(covered_items))
    partial = summary.get("partial", len(partial_items))
    missed = summary.get("missed", len(missed_items))
    
    def pct(n):
        return f"{n / total * 100:.1f}%" if total > 0 else "0%"
    
    lines.append(f"| ✅ 已覆盖 | {covered} | {pct(covered)} |")
    lines.append(f"| ⚠️ 部分覆盖 | {partial} | {pct(partial)} |")
    lines.append(f"| ❌ 未覆盖 | {missed} | {pct(missed)} |")
    lines.append("")
    
    # 按类型分组统计
    by_category = summary.get("by_category", [])
    if by_category:
        lines.append("## 按类型分组统计")
        lines.append("")
        lines.append("| 类型 | 总数 | 已覆盖 | 部分覆盖 | 未覆盖 | 覆盖率 |")
        lines.append("|------|------|--------|---------|--------|--------|")
        
        for cat in by_category:
            lines.append(
                f"| {cat['category']} | {cat['total']} | {cat['covered']} | "
                f"{cat['partial']} | {cat['missed']} | {cat['coverage_rate']:.1f}% |"
            )
        
        lines.append("")
    
    # 未覆盖项及建议
    if missed_items:
        lines.append("## 未覆盖项及补充建议")
        lines.append("")
        
        for i, item in enumerate(missed_items, 1):
            criterion = criteria_map.get(item.criterion_id)
            if not criterion:
                continue
            
            lines.append(f"### {i}. {criterion.test_point}")
            lines.append("")
            lines.append(f"**类型**: {criterion.category}  ")
            if criterion.test_content:
                lines.append(f"**测试内容**: {criterion.test_content}")
            lines.append("")
            
            # 显示相关需求
            if item.matched_requirements:
                lines.append("**📋 相关需求**:")
                for req in item.matched_requirements[:2]:
                    lines.append(f"> {req.get('text', '')[:200]}...")
                lines.append("")
            
            # 显示 AI 建议
            if item.llm_suggestion:
                lines.append("> 💡 **AI 建议**:")
                lines.append(">")
                # 尝试解析 JSON 格式的建议
                try:
                    import json
                    suggestion = json.loads(item.llm_suggestion)
                    for case in suggestion.get("suggested_cases", []):
                        lines.append(f"> **用例: {case.get('title', '未命名')}**")
                        for step in case.get("steps", []):
                            lines.append(f"> - {step}")
                        lines.append(f"> - 预期: {case.get('expected', '')}")
                        if case.get("requirement_ref"):
                            lines.append(f"> - 需求来源: {case['requirement_ref']}")
                        lines.append(">")
                except:
                    lines.append(f"> {item.llm_suggestion}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    # 部分覆盖项
    if partial_items:
        lines.append("## 部分覆盖项")
        lines.append("")
        lines.append("| 类型 | 测试点 | 匹配分数 | 匹配场景 |")
        lines.append("|------|--------|---------|---------|")
        
        for item in partial_items:
            criterion = criteria_map.get(item.criterion_id)
            if not criterion:
                continue
            
            scenarios = item.matched_scenarios[:2] if item.matched_scenarios else []
            scenario_text = ", ".join([s.get("title", "")[:30] for s in scenarios])
            
            llm_mark = " 🤖" if item.llm_verified else ""
            lines.append(
                f"| {criterion.category} | {criterion.test_point[:40]} | "
                f"{item.best_score:.2f}{llm_mark} | {scenario_text} |"
            )
        
        lines.append("")
    
    # 已覆盖项（按类型分组）
    if covered_items:
        lines.append("## 已覆盖项")
        lines.append("")
        
        # 按类型分组
        covered_by_cat = {}
        for item in covered_items:
            criterion = criteria_map.get(item.criterion_id)
            if not criterion:
                continue
            cat = criterion.category
            if cat not in covered_by_cat:
                covered_by_cat[cat] = []
            covered_by_cat[cat].append((criterion, item))
        
        for cat, items in covered_by_cat.items():
            lines.append(f"### {cat}")
            lines.append("")
            lines.append("| 测试点 | 匹配分数 | 匹配场景 |")
            lines.append("|--------|---------|---------|")
            
            for criterion, item in items:
                scenarios = item.matched_scenarios[:2] if item.matched_scenarios else []
                scenario_text = ", ".join([s.get("title", "")[:30] for s in scenarios])
                
                lines.append(
                    f"| {criterion.test_point[:50]} | {item.best_score:.2f} | {scenario_text} |"
                )
            
            lines.append("")
    
    # 页脚
    lines.append("---")
    lines.append("")
    lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    return "\n".join(lines)
