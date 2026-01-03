from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session


def create_job(db: Session, job_type: str) -> str:
    """创建任务并返回 job_id。"""

    job_id = str(uuid.uuid4())
    db.execute(
        text(
            """
            INSERT INTO coverage_platform.jobs(job_id, type, status, progress, result, error)
            VALUES (CAST(:job_id AS uuid), :type, 'running', 0, '{}'::jsonb, NULL)
            """
        ),
        {"job_id": job_id, "type": job_type},
    )
    db.commit()
    return job_id


def update_job(
    db: Session,
    job_id: str,
    *,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    """更新任务状态。"""

    sets = ["updated_at = now()"]
    params: Dict[str, Any] = {"job_id": job_id}

    if status is not None:
        sets.append("status = :status")
        params["status"] = status

    if progress is not None:
        sets.append("progress = :progress")
        params["progress"] = progress

    if result is not None:
        sets.append("result = CAST(:result AS jsonb)")
        params["result"] = json.dumps(result, ensure_ascii=False)

    if error is not None:
        sets.append("error = CAST(:error AS jsonb)")
        params["error"] = json.dumps(error, ensure_ascii=False)

    db.execute(
        text(
            f"""
            UPDATE coverage_platform.jobs
            SET {', '.join(sets)}
            WHERE job_id = CAST(:job_id AS uuid)
            """
        ),
        params,
    )
    db.commit()


def mark_succeeded(db: Session, job_id: str, result: Dict[str, Any]) -> None:
    update_job(db, job_id, status="succeeded", progress=1.0, result=result, error=None)


def mark_failed(db: Session, job_id: str, message: str, code: str = "INTERNAL_ERROR") -> None:
    update_job(
        db,
        job_id,
        status="failed",
        progress=1.0,
        error={"code": code, "message": message},
    )


def get_job(db: Session, job_id: str) -> Optional[Dict[str, Any]]:
    row = db.execute(
        text(
            """
            SELECT job_id::text AS job_id, type, status, progress,
                   to_char(created_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS created_at,
                   to_char(updated_at, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS updated_at,
                   result, error
            FROM coverage_platform.jobs
            WHERE job_id = CAST(:job_id AS uuid)
            """
        ),
        {"job_id": job_id},
    ).mappings().first()

    if not row:
        return None

    return {
        "job_id": row["job_id"],
        "type": row["type"],
        "status": row["status"],
        "progress": float(row["progress"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "result": row["result"] if isinstance(row["result"], dict) else {},
        "error": row["error"] if isinstance(row["error"], dict) else None,
    }
