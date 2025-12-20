from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.deps import db_session
from app.schemas import LinkConfirmRequest, OkResponse


router = APIRouter()


@router.post("/links/confirm", response_model=OkResponse)
def confirm_link(payload: LinkConfirmRequest, db: Session = Depends(db_session)):
    """人工确认或驳回关联关系（MVP）。"""

    # MVP：只记录到 kb_links.evidence 中，后续可扩展独立字段或审计表
    row = db.execute(
        text("SELECT link_id FROM coverage_platform.kb_links WHERE link_id = :id"),
        {"id": payload.link_id},
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="关联关系不存在")

    action = payload.action
    status = "covered" if action == "confirm" else "rejected"

    db.execute(
        text(
            """
            UPDATE coverage_platform.kb_links
            SET status = :status,
                evidence = jsonb_set(evidence, '{manual_review}', to_jsonb(:comment::text), true),
                updated_at = now()
            WHERE link_id = :id
            """
        ),
        {"status": status, "comment": payload.comment or "", "id": payload.link_id},
    )
    db.commit()

    return OkResponse(ok=True)
