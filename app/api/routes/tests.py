from __future__ import annotations

import hashlib
import uuid
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.deps import db_session
from app.schemas import (
    AcceptedJobResponse,
    TestScenario,
    TestsSearchItem,
    TestsSearchRequest,
    TestsSearchResponse,
)
from app.services.embedding import embed_text
from app.services.jobs_service import create_job, mark_failed, mark_succeeded, update_job
from app.services.xmind_parser import parse_xmind


router = APIRouter()


@router.post("/tests/import/xmind", response_model=AcceptedJobResponse)
def import_xmind(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    parse_mode: str = Form(default="leaf_only"),
    db: Session = Depends(db_session),
):
    """导入并索引 XMind 测试场景。

    MVP 范围：
    - 解包 .xmind（zip），优先解析 content.json。
    - 抽取叶子节点作为测试场景写入 tests_scenarios。
    - 为每个场景生成 embedding（OpenAI 可用时使用真实 embedding，否则退化为伪向量）。
    """

    job_id = create_job(db, "xmind_import")

    raw = file.file.read()
    file_name = file.filename or "unknown.xmind"
    file_hash = hashlib.sha256(raw).hexdigest()

    update_job(
        db,
        job_id,
        status="running",
        progress=0.01,
        result={"message": "已创建导入任务，后台执行中", "file_name": file_name, "file_hash": file_hash},
    )

    background_tasks.add_task(_run_xmind_import_job, job_id, raw, file_name, file_hash, parse_mode)
    return {"job_id": job_id}


def _run_xmind_import_job(job_id: str, raw: bytes, file_name: str, file_hash: str, parse_mode: str) -> None:
    db = SessionLocal()
    try:
        update_job(db, job_id, status="running", progress=0.05, result={"message": "开始解析 XMind", "file_hash": file_hash})

        # 创建/复用导入源（以 file_hash 去重）
        source_id = str(uuid.uuid4())
        source_row = db.execute(
            text(
                """
                INSERT INTO coverage_platform.tests_sources(source_id, source_type, file_name, file_hash)
                VALUES (CAST(:source_id AS uuid), 'xmind', :file_name, :file_hash)
                ON CONFLICT (file_hash)
                DO UPDATE SET file_name = EXCLUDED.file_name
                RETURNING source_id::text AS source_id
                """
            ),
            {"source_id": source_id, "file_name": file_name, "file_hash": file_hash},
        ).mappings().first()
        if not source_row:
            mark_failed(db, job_id, "创建 tests_sources 失败")
            return

        source_id = source_row["source_id"]

        try:
            nodes = parse_xmind(raw, parse_mode=parse_mode)
        except Exception as e:
            mark_failed(db, job_id, f"XMind 解析失败：{e}", code="UNPROCESSABLE")
            return

        if not nodes:
            mark_failed(db, job_id, "未解析到任何测试场景节点（请确认 XMind 版本/格式）", code="UNPROCESSABLE")
            return

        update_job(
            db,
            job_id,
            status="running",
            progress=0.25,
            result={"message": f"解析完成，准备入库节点数：{len(nodes)}", "source_id": source_id, "file_hash": file_hash},
        )

        inserted = 0
        for idx, n in enumerate(nodes, start=1):
            scenario_id = f"xmind:{file_hash}:{n.node_id}"
            title = n.title or "未命名场景"
            path = n.path or ""
            context_text = n.context_text
            emb = embed_text(context_text)

            db.execute(
                text(
                    """
                    INSERT INTO coverage_platform.tests_scenarios(
                      scenario_id, source_id, title, path, notes, context_text, embedding
                    )
                    VALUES(
                      :scenario_id, CAST(:source_id AS uuid), :title, :path, :notes, :context_text, :embedding
                    )
                    ON CONFLICT (scenario_id)
                    DO UPDATE SET
                      title = EXCLUDED.title,
                      path = EXCLUDED.path,
                      notes = EXCLUDED.notes,
                      context_text = EXCLUDED.context_text,
                      embedding = EXCLUDED.embedding
                    """
                ),
                {
                    "scenario_id": scenario_id,
                    "source_id": source_id,
                    "title": title,
                    "path": path,
                    "notes": n.notes,
                    "context_text": context_text,
                    "embedding": emb,
                },
            )
            inserted += 1

            if idx % 50 == 0:
                db.commit()
                progress = 0.25 + 0.7 * (idx / max(1, len(nodes)))
                update_job(
                    db,
                    job_id,
                    status="running",
                    progress=min(0.99, float(progress)),
                    result={"message": "入库中", "source_id": source_id, "file_hash": file_hash, "scenarios": inserted},
                )

        db.commit()
        mark_succeeded(
            db,
            job_id,
            {"message": "导入成功", "source_id": source_id, "file_hash": file_hash, "scenarios": inserted},
        )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        mark_failed(db, job_id, f"导入失败：{e}")
    finally:
        db.close()


@router.post("/tests/search", response_model=TestsSearchResponse)
def search_tests(payload: TestsSearchRequest, db: Session = Depends(db_session)):
    query_vec = embed_text(payload.query_text)
    # 将向量转换为 PostgreSQL vector 格式字符串
    vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

    filters_sql = "WHERE 1=1"
    params = {"q": vec_str, "k": payload.top_k}

    if payload.filters:
        if payload.filters.source_ids:
            filters_sql += " AND source_id = ANY(CAST(:source_ids AS uuid[]))"
            params["source_ids"] = payload.filters.source_ids
        if payload.filters.path_prefix:
            filters_sql += " AND path LIKE :path_prefix"
            params["path_prefix"] = payload.filters.path_prefix + "%"

    sql = text(
        f"""
        SELECT scenario_id, source_id::text AS source_id, title, path, notes, context_text,
               (1 - (embedding <=> CAST(:q AS vector))) AS score
        FROM coverage_platform.tests_scenarios
        {filters_sql}
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )

    rows = db.execute(sql, params).mappings().all()

    items: List[TestsSearchItem] = []
    for r in rows:
        scenario = TestScenario(
            scenario_id=r["scenario_id"],
            source_id=r["source_id"],
            title=r["title"],
            path=r["path"],
            notes=r["notes"],
            context_text=r["context_text"],
        )
        items.append(TestsSearchItem(scenario=scenario, score=float(r["score"])))

    return TestsSearchResponse(items=items)


@router.get("/tests/scenarios/{scenario_id}", response_model=TestScenario)
def get_scenario(scenario_id: str, db: Session = Depends(db_session)):
    r = db.execute(
        text(
            """
            SELECT scenario_id, source_id::text AS source_id, title, path, notes, context_text
            FROM coverage_platform.tests_scenarios
            WHERE scenario_id = :id
            """
        ),
        {"id": scenario_id},
    ).mappings().first()

    if not r:
        raise HTTPException(status_code=404, detail="测试场景不存在")

    return TestScenario(
        scenario_id=r["scenario_id"],
        source_id=r["source_id"],
        title=r["title"],
        path=r["path"],
        notes=r["notes"],
        context_text=r["context_text"],
    )


@router.get("/tests/sources")
def list_test_sources(db: Session = Depends(db_session)):
    """获取所有已导入的 XMind 来源列表。"""
    rows = db.execute(
        text(
            """
            SELECT source_id::text AS source_id, source_type, file_name, file_hash, imported_at
            FROM coverage_platform.tests_sources
            ORDER BY imported_at DESC
            """
        )
    ).mappings().all()
    
    items = [
        {
            "source_id": r["source_id"],
            "source_type": r["source_type"],
            "file_name": r["file_name"],
            "file_hash": r["file_hash"],
            "imported_at": str(r["imported_at"]),
        }
        for r in rows
    ]
    
    return {"items": items, "total": len(items)}

