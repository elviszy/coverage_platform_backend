from __future__ import annotations

import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector

from app.config import get_settings


settings = get_settings()


class Base(DeclarativeBase):
    """SQLAlchemy Declarative Base。"""

    pass


DB_SCHEMA = "coverage_platform"


class RequirementsPage(Base):
    """Confluence 页面元数据。"""

    __tablename__ = "requirements_pages"
    __table_args__ = {"schema": DB_SCHEMA}

    page_id: Mapped[str] = mapped_column(String, primary_key=True)
    page_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    body_storage: Mapped[str] = mapped_column(Text, nullable=False, default="")
    path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    labels: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    fetched_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    criteria: Mapped[list["RequirementCriterion"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
    )


class RequirementCriterion(Base):
    """验收标准（表格行级）。"""

    __tablename__ = "requirements_criteria"
    __table_args__ = {"schema": DB_SCHEMA}

    criterion_id: Mapped[str] = mapped_column(String, primary_key=True)
    page_id: Mapped[str] = mapped_column(String, ForeignKey(f"{DB_SCHEMA}.requirements_pages.page_id", ondelete="CASCADE"), nullable=False)
    page_version: Mapped[int] = mapped_column(Integer, nullable=False)
    page_url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False, default="")

    table_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    row_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    table_title: Mapped[str | None] = mapped_column(Text, nullable=True)

    headers: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    row_data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    page: Mapped[RequirementsPage] = relationship(back_populates="criteria")


class TestsSource(Base):
    """用例导入源（例如 XMind 文件）。"""

    __tablename__ = "tests_sources"
    __table_args__ = {"schema": DB_SCHEMA}

    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="xmind")
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(String, nullable=False)
    imported_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    scenarios: Mapped[list["TestScenario"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )


class TestScenario(Base):
    """测试场景（来自 XMind 节点）。"""

    __tablename__ = "tests_scenarios"
    __table_args__ = {"schema": DB_SCHEMA}

    scenario_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey(f"{DB_SCHEMA}.tests_sources.source_id", ondelete="CASCADE"), nullable=False)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_text: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dim), nullable=False)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    source: Mapped[TestsSource] = relationship(back_populates="scenarios")


class ReviewRun(Base):
    """一次评审任务（覆盖度 + 质量）。"""

    __tablename__ = "review_runs"
    __table_args__ = {"schema": DB_SCHEMA}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")

    requirements_scope: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tests_scope: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    finished_at: Mapped[str | None] = mapped_column(DateTime(timezone=True), nullable=True)


class KBLink(Base):
    """需求↔场景关联关系。"""

    __tablename__ = "kb_links"
    __table_args__ = {"schema": DB_SCHEMA}

    link_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey(f"{DB_SCHEMA}.review_runs.run_id", ondelete="SET NULL"), nullable=True)

    scenario_id: Mapped[str] = mapped_column(String, ForeignKey(f"{DB_SCHEMA}.tests_scenarios.scenario_id", ondelete="CASCADE"), nullable=False)
    criterion_id: Mapped[str] = mapped_column(String, ForeignKey(f"{DB_SCHEMA}.requirements_criteria.criterion_id", ondelete="CASCADE"), nullable=False)

    link_type: Mapped[str] = mapped_column(String, nullable=False, default="coverage")
    status: Mapped[str] = mapped_column(String, nullable=False, default="maybe")
    score_vector: Mapped[float] = mapped_column(Float, nullable=False)

    verifier_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    verifier_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class ReviewSummary(Base):
    """评审摘要（便于快速展示 /summary）。"""

    __tablename__ = "review_summary"
    __table_args__ = {"schema": DB_SCHEMA}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey(f"{DB_SCHEMA}.review_runs.run_id", ondelete="CASCADE"), primary_key=True)

    total_criteria: Mapped[int] = mapped_column(Integer, nullable=False)
    covered_criteria: Mapped[int] = mapped_column(Integer, nullable=False)
    coverage_rate: Mapped[float] = mapped_column(Float, nullable=False)

    module_breakdown: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    diversity_breakdown: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class QualityReviewItem(Base):
    """质量评审结果（按场景维度）。"""

    __tablename__ = "quality_review_items"
    __table_args__ = {"schema": DB_SCHEMA}

    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey(f"{DB_SCHEMA}.review_runs.run_id", ondelete="CASCADE"), primary_key=True)
    scenario_id: Mapped[str] = mapped_column(String, ForeignKey(f"{DB_SCHEMA}.tests_scenarios.scenario_id", ondelete="CASCADE"), primary_key=True)

    completeness_score: Mapped[int] = mapped_column(Integer, nullable=False)
    consistency_score: Mapped[int] = mapped_column(Integer, nullable=False)
    executable_score: Mapped[int] = mapped_column(Integer, nullable=False)

    issues: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    llm_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    llm_suggestions: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class Job(Base):
    """后台任务（建议对齐 /jobs/{job_id}）。"""

    __tablename__ = "jobs"
    __table_args__ = {"schema": DB_SCHEMA}

    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    result: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
