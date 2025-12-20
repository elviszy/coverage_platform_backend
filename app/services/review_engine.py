from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services.embedding import embed_text
from app.services.llm_verifier import verify_coverage_with_llm


settings = get_settings()


@dataclass
class CoverageLinkDraft:
    scenario_id: str
    criterion_id: str
    score_vector: float
    status: str  # covered | maybe | rejected
    verifier_used: bool
    verifier_reason: Optional[str]
    evidence: Dict[str, Any]


@dataclass
class ScenarioQuality:
    scenario_id: str
    completeness_score: int
    consistency_score: int
    executable_score: int
    issues: List[Dict[str, Any]]
    llm_used: bool
    llm_suggestions: Optional[Dict[str, Any]]


_KEYWORDS_EXPECTED = re.compile(r"应|预期|提示|返回|状态|跳转|成功|失败|错误码|展示|写入|拒绝|允许")
_KEYWORDS_INPUT = re.compile(r"输入|点击|选择|参数|为空|null|超长|越界|非法|重复|并发|超时")
_KEYWORDS_ACTION = re.compile(r"点击|提交|查询|创建|删除|更新|登录|登出|导出|上传|下载|修改")
_KEYWORDS_SECURITY = re.compile(r"越权|权限|鉴权|注入|XSS|CSRF|敏感|加密|脱敏|审计", re.IGNORECASE)


def _score_quality(scenario_id: str, title: str, path: str, notes: Optional[str], context_text: str) -> ScenarioQuality:
    """默认规则质量评分（不依赖 LLM）。"""

    text_all = " ".join([title or "", path or "", notes or "", context_text or ""]).strip()

    completeness = 0
    issues: List[Dict[str, Any]] = []

    if title and title.strip():
        completeness += 30
    else:
        issues.append({"type": "missing_title", "severity": "high", "message": "缺少标题"})

    if path and path.strip():
        completeness += 10
    else:
        issues.append({"type": "missing_path", "severity": "medium", "message": "缺少路径/模块信息"})

    if _KEYWORDS_EXPECTED.search(text_all):
        completeness += 25
    else:
        issues.append({"type": "missing_expected", "severity": "high", "message": "缺少可验证的预期结果描述"})

    if _KEYWORDS_INPUT.search(text_all):
        completeness += 20
    else:
        issues.append({"type": "missing_input", "severity": "medium", "message": "缺少输入/条件信息"})

    # 类型标签（正常/异常/边界/安全）粗略判断
    if any(k in text_all for k in ["正常", "异常", "边界", "安全"]):
        completeness += 15

    completeness = min(100, completeness)

    consistency = 0
    if "【" in title and "】" in title:
        consistency += 30
    else:
        issues.append({"type": "naming_format", "severity": "low", "message": "标题未使用推荐格式（例如【模块】【类型】场景描述）"})

    if any(k in title for k in ["正常", "异常", "边界", "安全"]):
        consistency += 20

    if 8 <= len(title) <= 50:
        consistency += 10

    if any(k in title for k in ["测试一下", "看看", "处理", "相关", "其它"]):
        consistency -= 20
        issues.append({"type": "ambiguous_title", "severity": "medium", "message": "标题存在含糊描述，建议补充动作与对象"})

    # 模块信息
    if path and path.strip() and (path.split("/")[-1] in title or path.split("/")[0] in title):
        consistency += 30

    consistency = max(0, min(100, consistency))

    executable = 0
    if _KEYWORDS_ACTION.search(text_all):
        executable += 25

    # 对象词典（MVP 简化）
    if any(k in text_all for k in ["页面", "按钮", "接口", "字段", "权限", "订单", "用户", "角色", "验证码", "登录"]):
        executable += 25

    if _KEYWORDS_EXPECTED.search(text_all):
        executable += 30

    if _KEYWORDS_INPUT.search(text_all) or _KEYWORDS_SECURITY.search(text_all):
        executable += 20

    executable = min(100, executable)

    return ScenarioQuality(
        scenario_id=scenario_id,
        completeness_score=completeness,
        consistency_score=consistency,
        executable_score=executable,
        issues=issues,
        llm_used=False,
        llm_suggestions=None,
    )


def _iter_scenarios(db: Session, tests_scope: dict) -> Iterable[Dict[str, Any]]:
    where = "WHERE 1=1"
    params: Dict[str, Any] = {}

    source_ids = tests_scope.get("source_ids")
    if isinstance(source_ids, list) and source_ids:
        where += " AND source_id = ANY(:source_ids::uuid[])"
        params["source_ids"] = source_ids

    path_prefix = tests_scope.get("path_prefix")
    if isinstance(path_prefix, str) and path_prefix.strip():
        where += " AND path LIKE :path_prefix"
        params["path_prefix"] = path_prefix.strip() + "%"

    rows = db.execute(
        text(
            f"""
            SELECT scenario_id, source_id::text AS source_id, title, path, notes, context_text
            FROM coverage_platform.tests_scenarios
            {where}
            """
        ),
        params,
    ).mappings().all()

    for r in rows:
        yield r


def _search_criteria(db: Session, query_vec: list, req_scope: dict, top_k: int) -> List[Dict[str, Any]]:
    where = "WHERE is_active = true"
    params: Dict[str, Any] = {"q": query_vec, "k": top_k}

    page_ids = req_scope.get("page_ids")
    if isinstance(page_ids, list) and page_ids:
        where += " AND page_id = ANY(:page_ids)"
        params["page_ids"] = page_ids

    path_prefix = req_scope.get("path_prefix")
    if isinstance(path_prefix, str) and path_prefix.strip():
        where += " AND path LIKE :path_prefix"
        params["path_prefix"] = path_prefix.strip() + "%"

    sql = text(
        f"""
        SELECT criterion_id, normalized_text, (1 - (embedding <=> :q)) AS score
        FROM coverage_platform.requirements_criteria
        {where}
        ORDER BY embedding <=> :q
        LIMIT :k
        """
    )

    return list(db.execute(sql, params).mappings().all())


def run_review(db: Session, run_id: str, payload: dict) -> Tuple[List[CoverageLinkDraft], List[ScenarioQuality]]:
    """执行一次评审：覆盖度 + 质量（规则版）。

    说明：
    - 覆盖度：对每个场景检索 topK 验收标准，按阈值判定 covered/maybe/rejected。
    - 可选 LLM verifier：仅对 maybe 的前 N 个候选做二次判定。
    - 质量：默认规则评分（不依赖 LLM）。
    """

    req_scope = (payload.get("requirements_scope") or {})
    tests_scope = (payload.get("tests_scope") or {})

    coverage_cfg = payload.get("coverage") or {}
    top_k = int(coverage_cfg.get("top_k") or 50)
    threshold_cover = float(coverage_cfg.get("threshold_cover") or 0.82)
    threshold_maybe = float(coverage_cfg.get("threshold_maybe") or 0.75)

    enable_llm_verifier = bool(coverage_cfg.get("enable_llm_verifier") or False)
    max_verify_per_scenario = int(coverage_cfg.get("max_verify_per_scenario") or settings.llm_max_verify_per_scenario)

    links: List[CoverageLinkDraft] = []
    qualities: List[ScenarioQuality] = []

    for s in _iter_scenarios(db, tests_scope):
        scenario_id = s["scenario_id"]
        title = s.get("title") or ""
        path = s.get("path") or ""
        notes = s.get("notes")
        context_text = s.get("context_text") or ""

        # 质量评分
        qualities.append(_score_quality(scenario_id, title, path, notes, context_text))

        # 覆盖检索
        query_vec = embed_text(context_text)
        candidates = _search_criteria(db, query_vec, req_scope, top_k)

        verified_count = 0
        for c in candidates:
            criterion_id = c["criterion_id"]
            criterion_text = c.get("normalized_text") or ""
            score = float(c.get("score") or 0.0)

            status = "rejected"
            if score >= threshold_cover:
                status = "covered"
            elif score >= threshold_maybe:
                status = "maybe"

            verifier_used = False
            verifier_reason: Optional[str] = None
            evidence: Dict[str, Any] = {
                "criterion_excerpt": criterion_text[:500],
                "scenario_excerpt": context_text[:500],
            }

            if status == "maybe" and enable_llm_verifier and verified_count < max_verify_per_scenario:
                verified_count += 1
                vr = verify_coverage_with_llm(
                    scenario_text=context_text,
                    criterion_text=criterion_text,
                    scenario_id=scenario_id,
                    criterion_id=criterion_id,
                )
                if vr:
                    verifier_used = True
                    verifier_reason = vr.reason
                    evidence.update(vr.evidence or {})
                    status = "covered" if vr.decision == "covered" else "rejected"

            links.append(
                CoverageLinkDraft(
                    scenario_id=scenario_id,
                    criterion_id=criterion_id,
                    score_vector=score,
                    status=status,
                    verifier_used=verifier_used,
                    verifier_reason=verifier_reason,
                    evidence=evidence,
                )
            )

    return links, qualities
