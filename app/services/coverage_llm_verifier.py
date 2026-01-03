"""覆盖度分析 LLM 验证器。

提供边界匹配二次验证和未覆盖项智能建议功能，支持需求上下文增强。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from app.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class CoverageVerifyResult:
    """边界匹配验证结果"""
    decision: str  # covered / partial / missed
    reason: str
    confidence: float


@dataclass
class MissSuggestionResult:
    """未覆盖项建议结果"""
    suggested_cases: List[Dict[str, Any]]
    priority: str  # high / medium / low
    notes: str


def _get_openai_client():
    """获取 OpenAI 客户端（如果已配置）"""
    try:
        from openai import OpenAI
        
        api_key = getattr(settings, 'openai_api_key', None)
        if not api_key:
            return None
        
        base_url = getattr(settings, 'openai_base_url', None)
        
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    except Exception as e:
        logger.warning(f"无法创建 OpenAI 客户端: {e}")
        return None


def is_llm_available() -> bool:
    """检查 LLM 是否可用"""
    return _get_openai_client() is not None


def verify_boundary_match_with_context(
    scenario_text: str,
    criterion_text: str,
    requirement_context: str,
    similarity_score: float,
) -> Optional[CoverageVerifyResult]:
    """
    对边界匹配进行 LLM 二次验证（结合需求上下文）。
    
    适用于相似度分数在 0.65~0.80 之间的边界情况。
    
    Args:
        scenario_text: 测试场景文本
        criterion_text: 公共测试标准文本
        requirement_context: 相关需求文档片段
        similarity_score: 语义相似度分数
        
    Returns:
        验证结果，如果 LLM 不可用返回 None
    """
    client = _get_openai_client()
    if not client:
        logger.debug("LLM 不可用，跳过边界验证")
        return None
    
    system_prompt = """你是一名测试评审专家。你的任务是判断给定的测试场景是否覆盖了指定的公共测试点。

判定时请参考需求文档中的相关描述，理解业务上下文。

判定标准：
- covered: 场景完全覆盖了测试点的核心内容
- partial: 场景部分覆盖，但缺少某些关键验证
- missed: 场景与测试点无关或覆盖极少

只输出 JSON：{"decision": "covered/partial/missed", "reason": "判定理由", "confidence": 0.0~1.0}"""

    user_prompt = f"""【公共测试点】
{criterion_text}

【测试场景】
{scenario_text}

【相关需求描述】
{requirement_context if requirement_context else "（无相关需求）"}

【语义相似度】{similarity_score:.2f}

请判断该场景是否覆盖了上述测试点。"""

    try:
        model = getattr(settings, 'openai_model_verifier', 'gpt-4o-mini')
        
        logger.info(f"[LLM-Verify] 开始边界验证 - 相似度: {similarity_score:.2f}")
        logger.info(f"[LLM-Verify] 模型: {model}")
        logger.info(f"[LLM-Verify] System Prompt: {system_prompt[:100]}...")
        logger.info(f"[LLM-Verify] User Prompt: {user_prompt}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"[LLM-Verify] 原始响应: {content}")
        
        # 解析 JSON
        # 处理可能的 markdown 代码块
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        
        logger.info(f"[LLM-Verify] 解析结果: decision={result.get('decision')}, confidence={result.get('confidence')}")
        
        return CoverageVerifyResult(
            decision=result.get("decision", "missed"),
            reason=result.get("reason", ""),
            confidence=float(result.get("confidence", 0.5)),
        )
        
    except Exception as e:
        logger.error(f"[LLM-Verify] 边界验证失败: {e}")
        return None


def generate_miss_suggestion_with_context(
    criterion_category: str,
    criterion_test_point: str,
    criterion_test_content: str,
    requirement_context: str,
) -> Optional[MissSuggestionResult]:
    """
    为未覆盖的公共测试点生成补充用例建议（结合需求上下文）。
    
    Args:
        criterion_category: 测试类型
        criterion_test_point: 测试点名称
        criterion_test_content: 测试内容描述
        requirement_context: 相关需求文档片段
        
    Returns:
        建议结果，如果 LLM 不可用返回 None
    """
    client = _get_openai_client()
    if not client:
        logger.debug("LLM 不可用，跳过建议生成")
        return None
    
    system_prompt = """你是一名资深测试工程师。根据给定的公共测试点和需求文档，生成具体的测试用例建议。

请结合需求文档中的业务细节，生成贴合实际业务场景的测试用例。

输出 JSON 格式：
{
    "suggested_cases": [
        {
            "title": "测试用例标题",
            "steps": ["步骤1", "步骤2", ...],
            "expected": "预期结果",
            "requirement_ref": "关联的需求描述"
        }
    ],
    "priority": "high/medium/low",
    "notes": "补充说明"
}"""

    user_prompt = f"""请为以下公共测试点生成补充用例建议：

【测试类型】{criterion_category}
【测试点】{criterion_test_point}
【测试内容】{criterion_test_content if criterion_test_content else "（无详细描述）"}

【相关需求描述】
{requirement_context if requirement_context else "（无相关需求）"}

请生成 1-2 个贴合业务场景的测试用例。"""

    try:
        model = getattr(settings, 'openai_model_verifier', 'gpt-4o-mini')
        
        logger.info(f"[LLM-Suggest] 开始生成建议 - 类型: {criterion_category}, 测试点: {criterion_test_point}")
        logger.info(f"[LLM-Suggest] 模型: {model}")
        logger.info(f"[LLM-Suggest] System Prompt: {system_prompt[:100]}...")
        logger.info(f"[LLM-Suggest] User Prompt: {user_prompt}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        logger.info(f"[LLM-Suggest] 原始响应: {content}")
        
        # 解析 JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        
        logger.info(f"[LLM-Suggest] 解析结果: {len(result.get('suggested_cases', []))} 个用例, priority={result.get('priority')}")
        
        return MissSuggestionResult(
            suggested_cases=result.get("suggested_cases", []),
            priority=result.get("priority", "medium"),
            notes=result.get("notes", ""),
        )
        
    except Exception as e:
        logger.error(f"[LLM-Suggest] 建议生成失败: {e}")
        return None


def batch_verify_boundaries(
    items: List[Dict[str, Any]],
    max_count: int = 20,
) -> Dict[str, CoverageVerifyResult]:
    """
    批量验证边界匹配项。
    
    Args:
        items: 边界项列表，每项包含 {criterion_id, scenario_text, criterion_text, requirement_context, score}
        max_count: 最多验证数量
        
    Returns:
        {criterion_id: CoverageVerifyResult, ...}
    """
    results = {}
    
    # 按分数排序，优先验证更接近阈值的项
    sorted_items = sorted(items, key=lambda x: abs(x["score"] - 0.725))[:max_count]
    
    for item in sorted_items:
        result = verify_boundary_match_with_context(
            scenario_text=item["scenario_text"],
            criterion_text=item["criterion_text"],
            requirement_context=item.get("requirement_context", ""),
            similarity_score=item["score"],
        )
        
        if result:
            results[item["criterion_id"]] = result
    
    return results


def batch_generate_suggestions(
    items: List[Dict[str, Any]],
    max_count: int = 10,
) -> Dict[str, MissSuggestionResult]:
    """
    批量生成未覆盖项建议。
    
    Args:
        items: 未覆盖项列表，每项包含 {criterion_id, category, test_point, test_content, requirement_context}
        max_count: 最多生成数量
        
    Returns:
        {criterion_id: MissSuggestionResult, ...}
    """
    results = {}
    
    # 取前 max_count 个
    for item in items[:max_count]:
        result = generate_miss_suggestion_with_context(
            criterion_category=item["category"],
            criterion_test_point=item["test_point"],
            criterion_test_content=item.get("test_content", ""),
            requirement_context=item.get("requirement_context", ""),
        )
        
        if result:
            results[item["criterion_id"]] = result
    
    return results
