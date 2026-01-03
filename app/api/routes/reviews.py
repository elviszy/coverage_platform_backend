from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.deps import db_session
from app.schemas import (
    AcceptedJobResponse,
    CoverageSummary,
    DiversityCount,
    GapItem,
    GapsRequest,
    GapsResponse,
    IssueCount,
    LinksResponse,
    Link,
    QualityIssuesRequest,
    QualityIssuesResponse,
    QualityReviewItem,
    QualitySummary,
    ReviewRun,
    ReviewRunListResponse,
    ReviewRunRequest,
    ReviewSummaryResponse,
    ExportRequest,
)
from app.services.jobs_service import create_job, mark_failed, mark_succeeded, update_job
from app.services.review_engine import run_review


router = APIRouter()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


_KW_EXCEPTION = re.compile(r"失败|错误|异常|拒绝|无效|超时|重试")
_KW_BOUNDARY = re.compile(r"边界|最大|最小|为空|null|0|极限|长度|上限|下限", re.IGNORECASE)
_KW_SECURITY = re.compile(r"越权|权限|鉴权|注入|XSS|CSRF|敏感|加密|脱敏|审计", re.IGNORECASE)


def _classify_scenario(text_all: str) -> str:
    if _KW_SECURITY.search(text_all):
        return "security"
    if _KW_BOUNDARY.search(text_all):
        return "boundary"
    if _KW_EXCEPTION.search(text_all):
        return "exception"
    return "normal"


def _priority_from_text(normalized_text: str) -> str:
    t = normalized_text or ""
    # 简单优先级规则（可后续配置化）
    if re.search(r"安全|权限|资金|支付|敏感|审计|不得", t):
        return "P0"
    if re.search(r"必须|应当|关键|核心|主流程", t):
        return "P1"
    return "P2"


@router.post("/reviews/runs", response_model=AcceptedJobResponse)
def create_review_run(payload: ReviewRunRequest, background_tasks: BackgroundTasks, db: Session = Depends(db_session)):
    """发起评审（异步后台执行）。"""

    job_id = create_job(db, "review_run")
    run_id = str(uuid.uuid4())
    payload_dict = payload.model_dump()

    db.execute(
        text(
            """
            INSERT INTO coverage_platform.review_runs(run_id, status, requirements_scope, tests_scope, config)
            VALUES (CAST(:run_id AS uuid), 'running', CAST(:req_scope AS jsonb), CAST(:test_scope AS jsonb), CAST(:config AS jsonb))
            """
        ),
        {
            "run_id": run_id,
            "req_scope": json.dumps(payload_dict.get("requirements_scope") or {}, ensure_ascii=False),
            "test_scope": json.dumps(payload_dict.get("tests_scope") or {}, ensure_ascii=False),
            "config": json.dumps(payload_dict, ensure_ascii=False),
        },
    )
    db.commit()

    update_job(
        db,
        job_id,
        status="running",
        progress=0.01,
        result={"run_id": run_id, "message": "评审已创建，后台执行中"},
    )

    background_tasks.add_task(_run_review_job, job_id, run_id, payload_dict)
    return {"job_id": job_id}


def _run_review_job(job_id: str, run_id: str, payload_dict: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        update_job(db, job_id, status="running", progress=0.05, result={"run_id": run_id, "message": "开始评审"})

        links, qualities = run_review(db, run_id=run_id, payload=payload_dict)
        update_job(
            db,
            job_id,
            status="running",
            progress=0.35,
            result={"run_id": run_id, "message": "已完成覆盖匹配与质量评分，开始写入结果"},
        )

        for l in links:
            db.execute(
                text(
                    """
                    INSERT INTO coverage_platform.kb_links(
                      run_id, scenario_id, criterion_id, link_type, status, score_vector,
                      verifier_used, verifier_reason, evidence
                    )
                    VALUES(
                      CAST(:run_id AS uuid), :scenario_id, :criterion_id, 'coverage', :status, :score_vector,
                      :verifier_used, :verifier_reason, CAST(:evidence AS jsonb)
                    )
                    ON CONFLICT (run_id, scenario_id, criterion_id)
                    DO UPDATE SET
                      status = EXCLUDED.status,
                      score_vector = EXCLUDED.score_vector,
                      verifier_used = EXCLUDED.verifier_used,
                      verifier_reason = EXCLUDED.verifier_reason,
                      evidence = EXCLUDED.evidence,
                      updated_at = now()
                    """
                ),
                {
                    "run_id": run_id,
                    "scenario_id": l.scenario_id,
                    "criterion_id": l.criterion_id,
                    "status": l.status,
                    "score_vector": l.score_vector,
                    "verifier_used": l.verifier_used,
                    "verifier_reason": l.verifier_reason,
                    "evidence": json.dumps(l.evidence or {}, ensure_ascii=False),
                },
            )

        for q in qualities:
            db.execute(
                text(
                    """
                    INSERT INTO coverage_platform.quality_review_items(
                      run_id, scenario_id, completeness_score, consistency_score, executable_score,
                      issues, llm_used, llm_suggestions
                    )
                    VALUES(
                      CAST(:run_id AS uuid), :scenario_id, :c, :s, :e,
                      CAST(:issues AS jsonb), :llm_used, CAST(:llm_suggestions AS jsonb)
                    )
                    ON CONFLICT (run_id, scenario_id)
                    DO UPDATE SET
                      completeness_score = EXCLUDED.completeness_score,
                      consistency_score = EXCLUDED.consistency_score,
                      executable_score = EXCLUDED.executable_score,
                      issues = EXCLUDED.issues,
                      llm_used = EXCLUDED.llm_used,
                      llm_suggestions = EXCLUDED.llm_suggestions
                    """
                ),
                {
                    "run_id": run_id,
                    "scenario_id": q.scenario_id,
                    "c": q.completeness_score,
                    "s": q.consistency_score,
                    "e": q.executable_score,
                    "issues": json.dumps(q.issues or [], ensure_ascii=False),
                    "llm_used": bool(q.llm_used),
                    "llm_suggestions": json.dumps(q.llm_suggestions or {}, ensure_ascii=False)
                    if q.llm_suggestions
                    else json.dumps(None),
                },
            )

        update_job(db, job_id, status="running", progress=0.75, result={"run_id": run_id, "message": "写入完成，计算摘要"})

        req_scope = payload_dict.get("requirements_scope") or {}
        where = "WHERE is_active = true"
        params: Dict[str, Any] = {}
        if isinstance(req_scope.get("page_ids"), list) and req_scope.get("page_ids"):
            where += " AND page_id = ANY(:page_ids)"
            params["page_ids"] = req_scope["page_ids"]
        if isinstance(req_scope.get("path_prefix"), str) and req_scope.get("path_prefix"):
            where += " AND path LIKE :path_prefix"
            params["path_prefix"] = req_scope["path_prefix"] + "%"

        total_criteria = int(
            db.execute(
                text(f"SELECT COUNT(*) FROM coverage_platform.requirements_criteria {where}"),
                params,
            ).scalar_one()
        )

        covered_criteria = int(
            db.execute(
                text(
                    """
                    SELECT COUNT(DISTINCT c.criterion_id)
                    FROM coverage_platform.requirements_criteria c
                    JOIN coverage_platform.kb_links l
                      ON l.criterion_id = c.criterion_id
                    WHERE l.run_id = CAST(:run_id AS uuid)
                      AND l.status = 'covered'
                      AND c.is_active = true
                    """
                ),
                {"run_id": run_id},
            ).scalar_one()
        )

        coverage_rate = (covered_criteria / total_criteria) if total_criteria else 0.0

        module_rows = db.execute(
            text(
                f"""
                SELECT path, COUNT(*) AS total
                FROM coverage_platform.requirements_criteria
                {where}
                GROUP BY path
                """
            ),
            params,
        ).mappings().all()

        module_breakdown: List[Dict[str, Any]] = []
        for mr in module_rows:
            path_key = mr["path"]
            total = int(mr["total"])
            covered = int(
                db.execute(
                    text(
                        """
                        SELECT COUNT(DISTINCT c.criterion_id)
                        FROM coverage_platform.requirements_criteria c
                        JOIN coverage_platform.kb_links l
                          ON l.criterion_id = c.criterion_id
                        WHERE l.run_id = CAST(:run_id AS uuid)
                          AND l.status = 'covered'
                          AND c.is_active = true
                          AND c.path = :path
                        """
                    ),
                    {"run_id": run_id, "path": path_key},
                ).scalar_one()
            )
            module_breakdown.append(
                {
                    "path": path_key,
                    "total": total,
                    "covered": covered,
                    "coverage_rate": (covered / total) if total else 0.0,
                }
            )

        diversity_counter = {"normal": 0, "exception": 0, "boundary": 0, "security": 0}
        scen_rows = list(
            db.execute(
                text(
                    """
                    SELECT title, path, notes, context_text
                    FROM coverage_platform.tests_scenarios
                    WHERE scenario_id IN (
                      SELECT DISTINCT scenario_id FROM coverage_platform.kb_links WHERE run_id = CAST(:run_id AS uuid)
                    )
                    """
                ),
                {"run_id": run_id},
            ).mappings().all()
        )
        for sr in scen_rows:
            t = " ".join(
                [
                    str(sr.get("title") or ""),
                    str(sr.get("path") or ""),
                    str(sr.get("notes") or ""),
                    str(sr.get("context_text") or ""),
                ]
            )
            diversity_counter[_classify_scenario(t)] += 1

        diversity_breakdown = [
            {"type": k, "count": int(v)} for k, v in diversity_counter.items() if v > 0
        ]

        db.execute(
            text(
                """
                INSERT INTO coverage_platform.review_summary(
                  run_id, total_criteria, covered_criteria, coverage_rate, module_breakdown, diversity_breakdown
                )
                VALUES(
                  CAST(:run_id AS uuid), :total, :covered, :rate, CAST(:modules AS jsonb), CAST(:diversity AS jsonb)
                )
                ON CONFLICT (run_id)
                DO UPDATE SET
                  total_criteria = EXCLUDED.total_criteria,
                  covered_criteria = EXCLUDED.covered_criteria,
                  coverage_rate = EXCLUDED.coverage_rate,
                  module_breakdown = EXCLUDED.module_breakdown,
                  diversity_breakdown = EXCLUDED.diversity_breakdown
                """
            ),
            {
                "run_id": run_id,
                "total": total_criteria,
                "covered": covered_criteria,
                "rate": coverage_rate,
                "modules": json.dumps(module_breakdown, ensure_ascii=False),
                "diversity": json.dumps(diversity_breakdown, ensure_ascii=False),
            },
        )

        db.execute(
            text("UPDATE coverage_platform.review_runs SET status='done', finished_at=now() WHERE run_id=CAST(:id AS uuid)"),
            {"id": run_id},
        )
        db.commit()

        mark_succeeded(
            db,
            job_id,
            {
                "run_id": run_id,
                "message": "评审完成",
                "total_criteria": total_criteria,
                "covered_criteria": covered_criteria,
            },
        )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass

        try:
            db.execute(
                text(
                    "UPDATE coverage_platform.review_runs SET status='failed', finished_at=now() WHERE run_id=CAST(:id AS uuid)"
                ),
                {"id": run_id},
            )
            db.commit()
        except Exception:
            pass

        mark_failed(db, job_id, f"评审执行失败：{e}")
    finally:
        db.close()


@router.get("/reviews/runs", response_model=ReviewRunListResponse)
def list_review_runs(page: int = 1, page_size: int = 20, db: Session = Depends(db_session)):
    offset = (page - 1) * page_size

    total = db.execute(text("SELECT COUNT(*) FROM coverage_platform.review_runs")).scalar_one()
    rows = db.execute(
        text(
            """
            SELECT run_id::text AS run_id, status, created_at
            FROM coverage_platform.review_runs
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
            """
        ),
        {"limit": page_size, "offset": offset},
    ).mappings().all()

    items: List[ReviewRun] = []
    for r in rows:
        items.append(
            ReviewRun(
                run_id=r["run_id"],
                status=r["status"],
                created_at=r["created_at"].strftime("%Y-%m-%dT%H:%M:%SZ") if r.get("created_at") else _now_iso(),
                config={},
            )
        )

    return ReviewRunListResponse(items=items, page=page, page_size=page_size, total=int(total))


@router.get("/reviews/runs/{run_id}/summary", response_model=ReviewSummaryResponse)
def get_summary(run_id: str, db: Session = Depends(db_session)):
    # summary
    row = db.execute(
        text(
            """
            SELECT total_criteria, covered_criteria, coverage_rate, module_breakdown, diversity_breakdown
            FROM coverage_platform.review_summary
            WHERE run_id = CAST(:id AS uuid)
            """
        ),
        {"id": run_id},
    ).mappings().first()

    if row:
        coverage = CoverageSummary(
            total_criteria=row["total_criteria"],
            covered_criteria=row["covered_criteria"],
            coverage_rate=float(row["coverage_rate"]),
            module_breakdown=[
                {
                    "path": x.get("path"),
                    "total": x.get("total"),
                    "covered": x.get("covered"),
                    "coverage_rate": x.get("coverage_rate"),
                }
                for x in (row.get("module_breakdown") or [])
                if isinstance(x, dict)
            ],
            diversity_breakdown=[
                {"type": x.get("type"), "count": x.get("count")}
                for x in (row.get("diversity_breakdown") or [])
                if isinstance(x, dict)
            ],
        )
    else:
        coverage = CoverageSummary(total_criteria=0, covered_criteria=0, coverage_rate=0.0)

    # quality summary
    qrow = db.execute(
        text(
            """
            SELECT
              AVG(completeness_score) AS avg_c,
              AVG(consistency_score) AS avg_s,
              AVG(executable_score) AS avg_e
            FROM coverage_platform.quality_review_items
            WHERE run_id = CAST(:id AS uuid)
            """
        ),
        {"id": run_id},
    ).mappings().first()

    avg_c = float(qrow.get("avg_c") or 0.0) if qrow else 0.0
    avg_s = float(qrow.get("avg_s") or 0.0) if qrow else 0.0
    avg_e = float(qrow.get("avg_e") or 0.0) if qrow else 0.0

    # issues breakdown（按 issues[].type 统计）
    issues_counter: Dict[str, int] = {}
    issue_rows = db.execute(
        text(
            """
            SELECT issues
            FROM coverage_platform.quality_review_items
            WHERE run_id = CAST(:id AS uuid)
            """
        ),
        {"id": run_id},
    ).mappings().all()
    for ir in issue_rows:
        issues = ir.get("issues")
        if isinstance(issues, list):
            for it in issues:
                if isinstance(it, dict) and isinstance(it.get("type"), str):
                    issues_counter[it["type"]] = issues_counter.get(it["type"], 0) + 1

    quality = QualitySummary(
        avg_completeness=avg_c,
        avg_consistency=avg_s,
        avg_executable=avg_e,
        issues_breakdown=[{"type": k, "count": v} for k, v in issues_counter.items()],
    )

    return ReviewSummaryResponse(coverage=coverage, quality=quality)


@router.post("/reviews/runs/{run_id}/gaps", response_model=GapsResponse)
def list_gaps(run_id: str, payload: GapsRequest, db: Session = Depends(db_session)):
    offset = (payload.page - 1) * payload.page_size

    # 未覆盖定义：在当前 run 内不存在 status='covered' 的 link
    total = int(
        db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM coverage_platform.requirements_criteria c
                WHERE c.is_active = true
                  AND NOT EXISTS (
                    SELECT 1 FROM coverage_platform.kb_links l
                    WHERE l.run_id = CAST(:rid AS uuid) AND l.criterion_id = c.criterion_id AND l.status = 'covered'
                  )
                """
            ),
            {"rid": run_id},
        ).scalar_one()
    )

    rows = db.execute(
        text(
            """
            SELECT criterion_id, page_id, page_url, page_version, path, table_idx, row_idx,
                   table_title, headers, row_data, normalized_text
            FROM coverage_platform.requirements_criteria c
            WHERE c.is_active = true
              AND NOT EXISTS (
                SELECT 1 FROM coverage_platform.kb_links l
                WHERE l.run_id = CAST(:rid AS uuid) AND l.criterion_id = c.criterion_id AND l.status = 'covered'
              )
            ORDER BY c.page_id, c.table_idx, c.row_idx
            LIMIT :limit OFFSET :offset
            """
        ),
        {"rid": run_id, "limit": payload.page_size, "offset": offset},
    ).mappings().all()

    items: List[GapItem] = []
    for r in rows:
        norm = r.get("normalized_text") or ""
        pr = _priority_from_text(norm)
        if pr not in payload.priority:
            continue

        criterion = {
            "criterion_id": r["criterion_id"],
            "page_id": r["page_id"],
            "page_url": r["page_url"],
            "page_version": r["page_version"],
            "path": r["path"],
            "table_idx": r["table_idx"],
            "row_idx": r["row_idx"],
            "table_title": r["table_title"],
            "headers": r["headers"] if isinstance(r.get("headers"), list) else [],
            "row": r["row_data"] if isinstance(r.get("row_data"), dict) else {},
            "normalized_text": norm,
            "is_active": True,
        }

        items.append(
            GapItem(
                priority=pr,
                criterion=criterion,  # pydantic 会自动校验
                suggested_queries=[],
            )
        )

    return GapsResponse(items=items, page=payload.page, page_size=payload.page_size, total=total)


@router.get("/reviews/runs/{run_id}/scenarios/{scenario_id}/links", response_model=LinksResponse)
def list_scenario_links(run_id: str, scenario_id: str, status: str | None = None, db: Session = Depends(db_session)):
    cond = "WHERE run_id = CAST(:rid AS uuid) AND scenario_id = :sid"
    params = {"rid": run_id, "sid": scenario_id}
    if status:
        cond += " AND status = :status"
        params["status"] = status

    rows = db.execute(
        text(
            f"""
            SELECT link_id::text AS link_id, run_id::text AS run_id, scenario_id, criterion_id,
                   link_type, status, score_vector, verifier_used, verifier_reason, evidence
            FROM coverage_platform.kb_links
            {cond}
            ORDER BY score_vector DESC
            """
        ),
        params,
    ).mappings().all()

    items: List[Link] = []
    for r in rows:
        items.append(
            Link(
                link_id=r["link_id"],
                run_id=r["run_id"],
                scenario_id=r["scenario_id"],
                criterion_id=r["criterion_id"],
                link_type=r["link_type"],
                status=r["status"],
                score_vector=float(r["score_vector"]),
                verifier_used=bool(r["verifier_used"]),
                verifier_reason=r["verifier_reason"],
                evidence=r["evidence"] if isinstance(r["evidence"], dict) else {},
            )
        )

    return LinksResponse(items=items)


@router.get("/reviews/runs/{run_id}/criteria/{criterion_id}/links", response_model=LinksResponse)
def list_criterion_links(run_id: str, criterion_id: str, status: str | None = None, db: Session = Depends(db_session)):
    cond = "WHERE run_id = CAST(:rid AS uuid) AND criterion_id = :cid"
    params = {"rid": run_id, "cid": criterion_id}
    if status:
        cond += " AND status = :status"
        params["status"] = status

    rows = db.execute(
        text(
            f"""
            SELECT link_id::text AS link_id, run_id::text AS run_id, scenario_id, criterion_id,
                   link_type, status, score_vector, verifier_used, verifier_reason, evidence
            FROM coverage_platform.kb_links
            {cond}
            ORDER BY score_vector DESC
            """
        ),
        params,
    ).mappings().all()

    items: List[Link] = []
    for r in rows:
        items.append(
            Link(
                link_id=r["link_id"],
                run_id=r["run_id"],
                scenario_id=r["scenario_id"],
                criterion_id=r["criterion_id"],
                link_type=r["link_type"],
                status=r["status"],
                score_vector=float(r["score_vector"]),
                verifier_used=bool(r["verifier_used"]),
                verifier_reason=r["verifier_reason"],
                evidence=r["evidence"] if isinstance(r["evidence"], dict) else {},
            )
        )

    return LinksResponse(items=items)


@router.post("/reviews/runs/{run_id}/quality/issues", response_model=QualityIssuesResponse)
def list_quality_issues(run_id: str, payload: QualityIssuesRequest, db: Session = Depends(db_session)):
    offset = (payload.page - 1) * payload.page_size

    where = "WHERE run_id = CAST(:rid AS uuid)"
    params: Dict[str, Any] = {"rid": run_id, "limit": payload.page_size, "offset": offset}

    if payload.filters:
        if payload.filters.min_executable_score is not None:
            where += " AND executable_score >= :min_exec"
            params["min_exec"] = payload.filters.min_executable_score
        if payload.filters.max_executable_score is not None:
            where += " AND executable_score <= :max_exec"
            params["max_exec"] = payload.filters.max_executable_score

    total = int(
        db.execute(text(f"SELECT COUNT(*) FROM coverage_platform.quality_review_items {where}"), params).scalar_one()
    )

    rows = db.execute(
        text(
            f"""
            SELECT scenario_id, completeness_score, consistency_score, executable_score, issues, llm_used, llm_suggestions
            FROM coverage_platform.quality_review_items
            {where}
            ORDER BY executable_score ASC
            LIMIT :limit OFFSET :offset
            """
        ),
        params,
    ).mappings().all()

    items: List[QualityReviewItem] = []
    issue_types = set(payload.filters.issue_types) if payload.filters and payload.filters.issue_types else None

    for r in rows:
        issues = r.get("issues") if isinstance(r.get("issues"), list) else []
        if issue_types is not None:
            issues = [it for it in issues if isinstance(it, dict) and it.get("type") in issue_types]

        items.append(
            QualityReviewItem(
                scenario_id=r["scenario_id"],
                completeness_score=int(r["completeness_score"]),
                consistency_score=int(r["consistency_score"]),
                executable_score=int(r["executable_score"]),
                issues=issues,
                llm_used=bool(r.get("llm_used")),
                llm_suggestions=r.get("llm_suggestions") if isinstance(r.get("llm_suggestions"), dict) else None,
            )
        )

    return QualityIssuesResponse(items=items, page=payload.page, page_size=payload.page_size, total=total)


@router.post("/reviews/runs/{run_id}/export", response_model=AcceptedJobResponse)
def export_review(run_id: str, payload: ExportRequest, db: Session = Depends(db_session)):
    """导出评审报告（MVP：返回 JSON/Markdown 内容到 job.result）。"""

    job_id = create_job(db, "export")

    # 读取 summary
    summary = get_summary(run_id, db)

    export_data: Dict[str, Any] = {
        "run_id": run_id,
        "summary": summary.model_dump(),
    }

    fmt = payload.format
    if fmt == "json":
        content = json.dumps(export_data, ensure_ascii=False, indent=2)
        mark_succeeded(db, job_id, {"message": "导出成功", "format": "json", "content": content})
    elif fmt == "md":
        cov = summary.coverage
        qua = summary.quality
        md = []
        md.append(f"# 评审报告\n")
        md.append(f"## 覆盖度\n")
        md.append(f"- 总验收标准数: {cov.total_criteria}\n")
        md.append(f"- 已覆盖数: {cov.covered_criteria}\n")
        md.append(f"- 覆盖率: {cov.coverage_rate:.4f}\n")
        md.append("\n## 质量\n")
        md.append(f"- 平均完善度: {qua.avg_completeness:.2f}\n")
        md.append(f"- 平均规范性: {qua.avg_consistency:.2f}\n")
        md.append(f"- 平均可执行性: {qua.avg_executable:.2f}\n")

        mark_succeeded(db, job_id, {"message": "导出成功", "format": "md", "content": "".join(md)})
    else:
        # xlsx 暂不实现（可后续补）
        mark_failed(db, job_id, "暂不支持 xlsx 导出，请使用 json 或 md", code="INVALID_INPUT")
        raise HTTPException(status_code=400, detail="暂不支持 xlsx 导出，请使用 json 或 md")

    return {"job_id": job_id}
