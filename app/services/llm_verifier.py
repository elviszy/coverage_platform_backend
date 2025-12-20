from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from app.config import get_settings


settings = get_settings()


# 进程内缓存（MVP）：避免同一对场景/验收标准反复调用 LLM
# key = (scenario_id, criterion_id, model)
_VERDICT_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


@dataclass
class VerifierResult:
    """LLM 覆盖判定结果。"""

    decision: str  # covered | rejected
    reason: str
    evidence: Dict[str, Any]


def verify_coverage_with_llm(
    scenario_text: str,
    criterion_text: str,
    scenario_id: str,
    criterion_id: str,
) -> Optional[VerifierResult]:
    """使用 OpenAI 对“场景是否覆盖验收标准”进行二次判定。

    说明：
    - 若未配置 OPENAI_API_KEY，则返回 None（表示跳过）。
    - 仅做最小实现：要求模型输出 JSON。
    """

    if not settings.openai_api_key:
        return None

    model = settings.openai_model_verifier
    cache_key = (scenario_id, criterion_id, model)
    cached = _VERDICT_CACHE.get(cache_key)
    if cached:
        return VerifierResult(
            decision=cached.get("decision", "rejected"),
            reason=cached.get("reason", ""),
            evidence=cached.get("evidence", {}) or {},
        )

    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    system = (
        "你是一名测试评审助手。你的任务是判断给定的测试场景是否能够覆盖指定的验收标准。"
        "只输出 JSON，不要输出额外文字。"
    )

    user = {
        "scenario": scenario_text,
        "acceptance_criterion": criterion_text,
        "output_format": {
            "decision": "covered 或 rejected",
            "reason": "简要说明原因",
            "evidence": {"matched": ["命中的关键词/字段"], "notes": "可选补充"},
        },
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content or ""

    # 宽松 JSON 解析：允许前后有杂质
    import json

    try:
        start = content.find("{")
        end = content.rfind("}")
        payload = json.loads(content[start : end + 1])
    except Exception:
        payload = {"decision": "rejected", "reason": "模型输出无法解析为 JSON", "evidence": {}}

    decision = payload.get("decision")
    if decision not in ("covered", "rejected"):
        decision = "rejected"

    result = {
        "decision": decision,
        "reason": str(payload.get("reason") or ""),
        "evidence": payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {},
    }

    _VERDICT_CACHE[cache_key] = result

    return VerifierResult(
        decision=result["decision"],
        reason=result["reason"],
        evidence=result["evidence"],
    )
