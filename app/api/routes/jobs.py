from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.deps import db_session
from app.services.jobs_service import get_job


router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job_api(job_id: str, db: Session = Depends(db_session)):
    """获取任务状态。"""

    job = get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")

    return job
