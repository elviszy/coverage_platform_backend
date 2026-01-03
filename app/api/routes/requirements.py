from __future__ import annotations

import hashlib
import json
import uuid
from collections import deque
from pathlib import Path
from typing import List, Set, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.deps import db_session
from app.schemas import (
    AcceptedJobResponse,
    ConfluenceImportRequest,
    RequirementsTextImportRequest,
    RequirementCriterion,
    RequirementsIndexRequest,
    RequirementsSearchRequest,
    RequirementsSearchResponse,
    RequirementsSearchItem,
)
from app.services.embedding import embed_text
from app.services.jobs_service import create_job, mark_failed, mark_succeeded, update_job
from app.services.confluence import (
    download_attachment,
    extract_page_id,
    fetch_attachments,
    fetch_child_page_ids,
    fetch_page_by_id,
)
from app.services.requirements_parser import parse_storage_to_criteria
from app.services.feature_extractor import extract_feature_points


router = APIRouter()


@router.post("/requirements/import/text", response_model=AcceptedJobResponse)
def import_requirements_text(
    background_tasks: BackgroundTasks,
    req: RequirementsTextImportRequest,
    db: Session = Depends(db_session),
):
    job_id = create_job(db, "requirements_import_text")

    page_id = str(uuid.uuid4())
    page_url = f"text://{page_id}"

    update_job(
        db,
        job_id,
        status="running",
        progress=0.01,
        result={"message": "已创建导入任务，后台执行中", "page_id": page_id, "page_url": page_url},
    )

    background_tasks.add_task(
        _run_requirements_text_import_job,
        job_id,
        {
            "page_id": page_id,
            "page_url": page_url,
            "title": req.title,
            "text": req.text,
            "path": req.path,
        },
    )

    return {"job_id": job_id}


def _run_requirements_text_import_job(job_id: str, payload: dict) -> None:
    db = SessionLocal()
    try:
        page_id = str(payload.get("page_id") or "")
        page_url = str(payload.get("page_url") or "")
        title = str(payload.get("title") or "")
        text_body = str(payload.get("text") or "")
        path = str(payload.get("path") or "")

        if not title:
            mark_failed(db, job_id, "title 不能为空", code="INVALID_INPUT")
            return

        version = 1
        # 用最简单的 HTML 包裹，复用现有解析/索引链路
        safe = (text_body or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body_storage = "\n".join([f"<p>{ln}</p>" for ln in safe.splitlines()])

        db.execute(
            text(
                """
                INSERT INTO coverage_platform.requirements_pages(page_id, page_url, title, version, body_storage, path, labels)
                VALUES (:page_id, :page_url, :title, :version, :body_storage, :path, CAST(:labels AS jsonb))
                ON CONFLICT (page_id)
                DO UPDATE SET
                  page_url = EXCLUDED.page_url,
                  title = EXCLUDED.title,
                  version = EXCLUDED.version,
                  body_storage = EXCLUDED.body_storage,
                  path = EXCLUDED.path,
                  labels = EXCLUDED.labels,
                  fetched_at = now()
                """
            ),
            {
                "page_id": page_id,
                "page_url": page_url,
                "title": title,
                "version": version,
                "body_storage": body_storage,
                "path": path,
                "labels": "{}",
            },
        )
        db.commit()
        mark_succeeded(db, job_id, {"message": "导入成功", "page_id": page_id})
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        mark_failed(db, job_id, f"导入失败：{e}")
    finally:
        db.close()


@router.post("/requirements/import/docx", response_model=AcceptedJobResponse)
def import_requirements_docx(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(default=""),
    path: str = Form(default=""),
    db: Session = Depends(db_session),
):
    job_id = create_job(db, "requirements_import_docx")

    raw = file.file.read()
    file_name = file.filename or "requirements.docx"
    file_hash = hashlib.sha256(raw).hexdigest()

    page_id = str(uuid.uuid4())
    page_url = f"docx://{file_hash}"

    update_job(
        db,
        job_id,
        status="running",
        progress=0.01,
        result={"message": "已创建导入任务，后台执行中", "page_id": page_id, "file_name": file_name, "file_hash": file_hash},
    )

    background_tasks.add_task(
        _run_requirements_docx_import_job,
        job_id,
        {
            "page_id": page_id,
            "page_url": page_url,
            "title": title,
            "path": path,
            "file_name": file_name,
            "file_hash": file_hash,
        },
        raw,
    )

    return {"job_id": job_id}


def _run_requirements_docx_import_job(job_id: str, meta: dict, raw: bytes) -> None:
    db = SessionLocal()
    try:
        page_id = str(meta.get("page_id") or "")
        page_url = str(meta.get("page_url") or "")
        title = str(meta.get("title") or "")
        path = str(meta.get("path") or "")
        file_name = str(meta.get("file_name") or "")

        update_job(db, job_id, status="running", progress=0.05, result={"message": "开始解析 DOCX", "page_id": page_id})

        # 延迟导入，避免未安装 python-docx 时影响其它功能
        from docx import Document
        import io

        doc = Document(io.BytesIO(raw))
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]

        if not title:
            title = file_name or "Word 需求"

        version = 1
        safe_lines = [
            (ln or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            for ln in paras
        ]
        body_storage = "\n".join([f"<p>{ln}</p>" for ln in safe_lines])

        db.execute(
            text(
                """
                INSERT INTO coverage_platform.requirements_pages(page_id, page_url, title, version, body_storage, path, labels)
                VALUES (:page_id, :page_url, :title, :version, :body_storage, :path, CAST(:labels AS jsonb))
                ON CONFLICT (page_id)
                DO UPDATE SET
                  page_url = EXCLUDED.page_url,
                  title = EXCLUDED.title,
                  version = EXCLUDED.version,
                  body_storage = EXCLUDED.body_storage,
                  path = EXCLUDED.path,
                  labels = EXCLUDED.labels,
                  fetched_at = now()
                """
            ),
            {
                "page_id": page_id,
                "page_url": page_url,
                "title": title,
                "version": version,
                "body_storage": body_storage,
                "path": path,
                "labels": "{}",
            },
        )
        db.commit()
        mark_succeeded(db, job_id, {"message": "导入成功", "page_id": page_id})
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        mark_failed(db, job_id, f"导入失败：{e}")
    finally:
        db.close()

@router.post("/requirements/import/confluence", response_model=AcceptedJobResponse)
def import_confluence(req: ConfluenceImportRequest, background_tasks: BackgroundTasks, db: Session = Depends(db_session)):
    """导入 Confluence 页面。

    MVP 范围：
    - 支持输入单个页面链接（含 pageId）。
    - 通过 Confluence Data Center REST API 拉取 body.storage/version/labels/ancestors。
    - 将页面原文写入 requirements_pages.body_storage。
    """

    job_id = create_job(db, "confluence_import")

    page_id = extract_page_id(req.page_url)
    if not page_id:
        mark_failed(db, job_id, "无法从链接中解析 pageId，请确认链接格式", code="INVALID_INPUT")
        raise HTTPException(status_code=400, detail="无法从链接中解析 pageId，请确认链接格式")

    update_job(
        db,
        job_id,
        status="running",
        progress=0.01,
        result={
            "message": "已创建导入任务，后台执行中",
            "root_page_id": page_id,
            "recursive": req.recursive,
            "max_depth": req.max_depth,
            "include_attachments": req.include_attachments,
        },
    )

    background_tasks.add_task(
        _run_confluence_import_job,
        job_id,
        {
            "page_id": page_id,
            "recursive": req.recursive,
            "max_depth": req.max_depth,
            "include_attachments": req.include_attachments,
        },
    )

    return {"job_id": job_id}


def _run_confluence_import_job(job_id: str, payload: dict) -> None:
    db = SessionLocal()
    try:
        page_id = str(payload.get("page_id") or "")
        recursive = bool(payload.get("recursive") or False)
        max_depth = int(payload.get("max_depth") or 0)
        include_attachments = bool(payload.get("include_attachments") or False)

        def upsert_page(p) -> None:
            path = "/".join([x for x in (p.ancestors + [p.title]) if x])
            labels_json = p.labels

            db.execute(
                text(
                    """
                    INSERT INTO coverage_platform.requirements_pages(page_id, page_url, title, version, body_storage, path, labels)
                    VALUES (:page_id, :page_url, :title, :version, :body_storage, :path, CAST(:labels AS jsonb))
                    ON CONFLICT (page_id)
                    DO UPDATE SET
                      page_url = EXCLUDED.page_url,
                      title = EXCLUDED.title,
                      version = EXCLUDED.version,
                      body_storage = EXCLUDED.body_storage,
                      path = EXCLUDED.path,
                      labels = EXCLUDED.labels,
                      fetched_at = now()
                    """
                ),
                {
                    "page_id": p.page_id,
                    "page_url": p.page_url,
                    "title": p.title,
                    "version": p.version,
                    "body_storage": p.body_storage,
                    "path": path,
                    "labels": json.dumps(labels_json, ensure_ascii=False),
                },
            )

        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque()
        queue.append((page_id, 0))

        imported_pages: List[str] = []
        errors: List[str] = []
        imported_attachments = 0

        update_job(db, job_id, status="running", progress=0.05, result={"message": "开始导入", "root_page_id": page_id})

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            try:
                page = fetch_page_by_id(current_id)
                upsert_page(page)
                db.commit()
                imported_pages.append(current_id)
            except Exception as e:
                db.rollback()
                errors.append(f"{current_id}:{e}")

            if include_attachments:
                try:
                    atts = fetch_attachments(current_id)
                    for a in atts:
                        try:
                            out_dir = Path(__file__).resolve().parents[3] / "data" / "confluence_attachments" / str(current_id)
                            file_path = download_attachment(a, out_dir=out_dir)

                            db.execute(
                                text(
                                    """
                                    INSERT INTO coverage_platform.requirements_attachments(
                                      page_id, attachment_id, filename, media_type, file_path, download_url
                                    )
                                    VALUES(
                                      :page_id, :attachment_id, :filename, :media_type, :file_path, :download_url
                                    )
                                    ON CONFLICT (page_id, attachment_id)
                                    DO UPDATE SET
                                      filename = EXCLUDED.filename,
                                      media_type = EXCLUDED.media_type,
                                      file_path = EXCLUDED.file_path,
                                      download_url = EXCLUDED.download_url,
                                      created_at = now()
                                    """
                                ),
                                {
                                    "page_id": str(current_id),
                                    "attachment_id": a.attachment_id,
                                    "filename": a.filename,
                                    "media_type": a.media_type,
                                    "file_path": str(file_path),
                                    "download_url": a.download_url,
                                },
                            )
                            db.commit()
                            imported_attachments += 1
                            
                            # RAG 附件处理：将附件纳入知识图谱
                            try:
                                import asyncio
                                from app.services.rag_service import get_rag_service
                                
                                rag_service = get_rag_service()
                                
                                async def process_att():
                                    await rag_service.initialize()
                                    await rag_service.process_attachment(str(file_path))
                                
                                asyncio.get_event_loop().run_until_complete(process_att())
                            except Exception as rag_err:
                                # RAG 处理失败不影响主导入流程
                                print(f"[confluence_import] RAG 附件处理警告: {rag_err}")
                                
                        except Exception as e:
                            db.rollback()
                            errors.append(f"attachment_{current_id}:{a.attachment_id}:{e}")
                except Exception as e:
                    errors.append(f"attachments_of_{current_id}:{e}")

            if recursive and depth < max_depth:
                try:
                    child_ids = fetch_child_page_ids(current_id)
                    for cid in child_ids:
                        if cid not in visited:
                            queue.append((cid, depth + 1))
                except Exception as e:
                    errors.append(f"children_of_{current_id}:{e}")

            done = len(visited)
            pending = len(queue)
            denom = done + pending
            progress = (done / denom) if denom else 0.0
            update_job(
                db,
                job_id,
                status="running",
                progress=max(0.05, min(0.99, float(progress))),
                result={
                    "root_page_id": page_id,
                    "recursive": recursive,
                    "max_depth": max_depth,
                    "imported_pages": len(imported_pages),
                    "imported_attachments": imported_attachments,
                    "errors": errors[-20:],
                },
            )

        if not imported_pages:
            mark_failed(db, job_id, "导入失败：未成功导入任何页面")
            return

        mark_succeeded(
            db,
            job_id,
            {
                "message": "导入成功",
                "root_page_id": page_id,
                "recursive": recursive,
                "max_depth": max_depth,
                "imported_pages": len(imported_pages),
                "imported_attachments": imported_attachments,
                "errors": errors,
            },
        )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        mark_failed(db, job_id, f"导入失败：{e}")
    finally:
        db.close()


@router.post("/requirements/index", response_model=AcceptedJobResponse)
def index_requirements(req: RequirementsIndexRequest, background_tasks: BackgroundTasks, db: Session = Depends(db_session)):
    """解析并索引验收标准表格行到需求知识库。

    MVP 范围：
    - 从 requirements_pages.body_storage 解析表格，并按“表格行级”生成 requirements_criteria。
    - 对每行生成 normalized_text，并计算 embedding。
    """

    job_id = create_job(db, "requirements_index")

    # 轻量预检查：保持原先 404 语义
    where = "WHERE 1=1"
    params: dict = {}
    if req.scope.page_ids:
        where += " AND page_id = ANY(:page_ids)"
        params["page_ids"] = req.scope.page_ids

    page_count = int(
        db.execute(
            text(f"SELECT COUNT(*) FROM coverage_platform.requirements_pages {where}"),
            params,
        ).scalar_one()
    )
    if page_count <= 0:
        mark_failed(db, job_id, "未找到可索引的页面，请先导入 Confluence 页面", code="NOT_FOUND")
        raise HTTPException(status_code=404, detail="未找到可索引的页面，请先导入 Confluence 页面")

    update_job(
        db,
        job_id,
        status="running",
        progress=0.01,
        result={"message": "已创建索引任务，后台执行中", "pages": page_count},
    )
    background_tasks.add_task(
        _run_requirements_index_job,
        job_id,
        req.model_dump(),
    )
    return {"job_id": job_id}


def _run_requirements_index_job(job_id: str, payload: dict) -> None:
    db = SessionLocal()
    try:
        scope = (payload.get("scope") or {})
        page_ids = scope.get("page_ids")
        only_latest = bool(scope.get("only_latest") if scope.get("only_latest") is not None else True)

        where = "WHERE 1=1"
        params: dict = {}
        if isinstance(page_ids, list) and page_ids:
            where += " AND page_id = ANY(:page_ids)"
            params["page_ids"] = page_ids

        pages = db.execute(
            text(
                f"""
                SELECT page_id, page_url, title, version, body_storage, path
                FROM coverage_platform.requirements_pages
                {where}
                """
            ),
            params,
        ).mappings().all()

        if not pages:
            mark_failed(db, job_id, "未找到可索引的页面，请先导入 Confluence 页面", code="NOT_FOUND")
            return

        update_job(db, job_id, status="running", progress=0.05, result={"message": f"开始索引页面数：{len(pages)}"})

        total_rows = 0
        for idx, p in enumerate(pages, start=1):
            page_id = p["page_id"]
            title = p["title"]
            page_url = p["page_url"]
            version = int(p["version"])
            body_storage = p["body_storage"] or ""
            path = p.get("path") or ""

            if only_latest:
                db.execute(
                    text(
                        """
                        UPDATE coverage_platform.requirements_criteria
                        SET is_active = false
                        WHERE page_id = :page_id
                        """
                    ),
                    {"page_id": page_id},
                )

            rows = parse_storage_to_criteria(
                page_id=page_id,
                page_url=page_url,
                title=title,
                page_version=version,
                storage_html=body_storage,
                path=path,
            )

            for r in rows:
                emb = embed_text(r.normalized_text)
                # 提取需求点/功能点（返回 Markdown 文本）
                feature_points_md = extract_feature_points(r.normalized_text)
                
                db.execute(
                    text(
                        """
                        INSERT INTO coverage_platform.requirements_criteria(
                          criterion_id, page_id, page_version, page_url, title, path,
                          table_idx, row_idx, table_title, headers, row_data, normalized_text, embedding, is_active, feature_points
                        )
                        VALUES(
                          :criterion_id, :page_id, :page_version, :page_url, :title, :path,
                          :table_idx, :row_idx, :table_title, CAST(:headers AS jsonb), CAST(:row_data AS jsonb), :normalized_text, :embedding, true, :feature_points
                        )
                        ON CONFLICT (criterion_id)
                        DO UPDATE SET
                          page_version = EXCLUDED.page_version,
                          page_url = EXCLUDED.page_url,
                          title = EXCLUDED.title,
                          path = EXCLUDED.path,
                          table_title = EXCLUDED.table_title,
                          headers = EXCLUDED.headers,
                          row_data = EXCLUDED.row_data,
                          normalized_text = EXCLUDED.normalized_text,
                          embedding = EXCLUDED.embedding,
                          is_active = true,
                          feature_points = EXCLUDED.feature_points
                        """
                    ),
                    {
                        "criterion_id": r.criterion_id,
                        "page_id": page_id,
                        "page_version": version,
                        "page_url": page_url,
                        "title": title,
                        "path": r.path,
                        "table_idx": r.table_idx,
                        "row_idx": r.row_idx,
                        "table_title": r.table_title,
                        "headers": "[]" if r.headers is None else __import__("json").dumps(r.headers, ensure_ascii=False),
                        "row_data": __import__("json").dumps(r.row_data, ensure_ascii=False),
                        "normalized_text": r.normalized_text,
                        "embedding": emb,
                        "feature_points": feature_points_md,
                    },
                )
                total_rows += 1

            db.commit()

            progress = 0.05 + 0.9 * (idx / max(1, len(pages)))
            update_job(
                db,
                job_id,
                status="running",
                progress=min(0.99, float(progress)),
                result={"message": "索引进行中", "indexed_rows": total_rows, "processed_pages": idx, "total_pages": len(pages)},
            )

        # 集成 RAG 知识图谱：将索引的内容同步插入知识图谱
        try:
            import asyncio
            from app.services.rag_service import get_rag_service
            
            rag_service = get_rag_service()
            
            async def sync_to_rag():
                await rag_service.initialize()
                # 查询刚索引的验收标准
                criteria_rows = db.execute(
                    text(
                        """
                        SELECT criterion_id, normalized_text, path, table_title
                        FROM coverage_platform.requirements_criteria
                        WHERE is_active = true
                        """
                    )
                ).mappings().all()
                
                for cr in criteria_rows:
                    await rag_service.insert_requirement(
                        text=cr["normalized_text"],
                        metadata={
                            "criterion_id": cr["criterion_id"],
                            "path": cr["path"],
                            "table_title": cr["table_title"],
                        }
                    )
            
            asyncio.get_event_loop().run_until_complete(sync_to_rag())
        except Exception as rag_err:
            # RAG 同步失败不影响主流程
            print(f"[requirements/index] RAG 同步警告: {rag_err}")

        mark_succeeded(db, job_id, {"message": "索引完成", "indexed_rows": total_rows})
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        mark_failed(db, job_id, f"索引失败：{e}")
    finally:
        db.close()


@router.post("/requirements/search", response_model=RequirementsSearchResponse)
def search_requirements(payload: RequirementsSearchRequest, db: Session = Depends(db_session)):
    """检索需求知识库（pgvector）。"""

    query_vec = embed_text(payload.query_text)
    # 将向量转换为 PostgreSQL vector 格式字符串
    vec_str = "[" + ",".join(str(x) for x in query_vec) + "]"

    # 说明：
    # - 使用 pgvector 的 cosine distance：embedding <=> query_vec
    # - 相似度 = 1 - distance
    # - 这里用原生 SQL，避免不同 SQLAlchemy/pgvector 版本差异导致的兼容问题
    filters_sql = "WHERE is_active = true"
    params = {"q": vec_str, "k": payload.top_k}

    if payload.filters:
        if payload.filters.only_active is False:
            filters_sql = "WHERE 1=1"
        if payload.filters.page_ids:
            filters_sql += " AND page_id = ANY(:page_ids)"
            params["page_ids"] = payload.filters.page_ids
        if payload.filters.path_prefix:
            filters_sql += " AND path LIKE :path_prefix"
            params["path_prefix"] = payload.filters.path_prefix + "%"

    sql = text(
        f"""
        SELECT
          criterion_id, page_id, page_url, page_version, path, table_idx, row_idx,
          table_title, headers, row_data, normalized_text, is_active, feature_points,
          (1 - (embedding <=> CAST(:q AS vector))) AS score
        FROM coverage_platform.requirements_criteria
        {filters_sql}
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
        """
    )

    rows = db.execute(sql, params).mappings().all()

    items: List[RequirementsSearchItem] = []
    for r in rows:
        # feature_points 现在是 Markdown 字符串
        fp = r.get("feature_points")
        feature_points = fp if isinstance(fp, str) else ""
        
        criterion = RequirementCriterion(
            criterion_id=r["criterion_id"],
            page_id=r["page_id"],
            page_url=r["page_url"],
            page_version=r["page_version"],
            path=r["path"],
            table_idx=r["table_idx"],
            row_idx=r["row_idx"],
            table_title=r["table_title"],
            headers=r["headers"] if isinstance(r["headers"], list) else [],
            row=r["row_data"] if isinstance(r["row_data"], dict) else {},
            normalized_text=r["normalized_text"],
            is_active=r["is_active"],
            feature_points=feature_points,
        )
        items.append(RequirementsSearchItem(criterion=criterion, score=float(r["score"])))

    return RequirementsSearchResponse(items=items)


@router.get("/requirements/criteria/{criterion_id}", response_model=RequirementCriterion)
def get_criterion(criterion_id: str, db: Session = Depends(db_session)):
    sql = text(
        """
        SELECT criterion_id, page_id, page_url, page_version, path, table_idx, row_idx,
               table_title, headers, row_data, normalized_text, is_active
        FROM coverage_platform.requirements_criteria
        WHERE criterion_id = :id
        """
    )
    r = db.execute(sql, {"id": criterion_id}).mappings().first()
    if not r:
        raise HTTPException(status_code=404, detail="验收标准不存在")

    return RequirementCriterion(
        criterion_id=r["criterion_id"],
        page_id=r["page_id"],
        page_url=r["page_url"],
        page_version=r["page_version"],
        path=r["path"],
        table_idx=r["table_idx"],
        row_idx=r["row_idx"],
        table_title=r["table_title"],
        headers=r["headers"] if isinstance(r["headers"], list) else [],
        row=r["row_data"] if isinstance(r["row_data"], dict) else {},
        normalized_text=r["normalized_text"],
        is_active=r["is_active"],
    )


@router.get("/requirements/pages")
def list_requirements_pages(db: Session = Depends(db_session)):
    """获取所有已导入的需求页面列表。"""
    rows = db.execute(
        text(
            """
            SELECT page_id, page_url, title, version, path, fetched_at
            FROM coverage_platform.requirements_pages
            ORDER BY fetched_at DESC
            """
        )
    ).mappings().all()
    
    items = [
        {
            "page_id": r["page_id"],
            "page_url": r["page_url"],
            "title": r["title"],
            "version": r["version"],
            "path": r["path"],
            "fetched_at": str(r["fetched_at"]),
        }
        for r in rows
    ]
    
    return {"items": items, "total": len(items)}

