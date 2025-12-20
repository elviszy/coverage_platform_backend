from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ------------------- 通用 -------------------


class ErrorObject(BaseModel):
    code: str = Field(description="错误码")
    message: str = Field(description="错误描述")
    details: Dict[str, Any] = Field(default_factory=dict, description="错误详情")


class ErrorResponse(BaseModel):
    error: ErrorObject


class AcceptedJobResponse(BaseModel):
    job_id: str


JobType = Literal["confluence_import", "requirements_index", "xmind_import", "review_run", "export"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]


class Job(BaseModel):
    job_id: str
    type: JobType
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    created_at: str
    updated_at: str
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ErrorObject] = None


# ------------------- 需求库 -------------------


class ConfluenceImportRequest(BaseModel):
    page_url: str = Field(description="Confluence 页面链接")
    recursive: bool = Field(default=False, description="是否递归导入子页面")
    max_depth: int = Field(default=0, ge=0, description="递归最大深度")
    include_attachments: bool = Field(default=True, description="是否拉取附件")


class RequirementsTextImportRequest(BaseModel):
    title: str = Field(description="需求标题")
    text: str = Field(description="需求纯文本内容")
    path: str = Field(default="", description="可选：路径/分类")


class RequirementsScope(BaseModel):
    page_ids: Optional[List[str]] = Field(default=None, description="限定页面 ID 列表")
    path_prefix: Optional[str] = Field(default=None, description="路径前缀过滤")
    only_latest: bool = Field(default=True, description="是否仅处理最新版本")


class EmbeddingConfig(BaseModel):
    dim: int = Field(default=1536, description="向量维度")
    model: str = Field(default="text-embedding-3-small", description="embedding 模型")


class RequirementsIndexRequest(BaseModel):
    scope: RequirementsScope
    embedding: Optional[EmbeddingConfig] = None
    reindex: bool = Field(default=False, description="是否强制重建索引")


class RequirementsSearchFilters(BaseModel):
    page_ids: Optional[List[str]] = None
    path_prefix: Optional[str] = None
    only_active: bool = True


class RequirementsSearchRequest(BaseModel):
    query_text: str
    top_k: int = Field(default=20, ge=1, le=200)
    filters: Optional[RequirementsSearchFilters] = None


class RequirementCriterion(BaseModel):
    criterion_id: str
    page_id: str
    page_url: str
    page_version: int
    path: str
    table_idx: int
    row_idx: int
    table_title: Optional[str] = None
    headers: List[str]
    row: Dict[str, Any]
    normalized_text: str
    is_active: bool = True


class RequirementsSearchItem(BaseModel):
    criterion: RequirementCriterion
    score: float


class RequirementsSearchResponse(BaseModel):
    items: List[RequirementsSearchItem]


# ------------------- 用例库 -------------------


class TestsSearchFilters(BaseModel):
    source_ids: Optional[List[str]] = None
    path_prefix: Optional[str] = None


class TestsSearchRequest(BaseModel):
    query_text: str
    top_k: int = Field(default=20, ge=1, le=200)
    filters: Optional[TestsSearchFilters] = None


class TestScenario(BaseModel):
    scenario_id: str
    source_id: str
    title: str
    path: str
    notes: Optional[str] = None
    context_text: str


class TestsSearchItem(BaseModel):
    scenario: TestScenario
    score: float


class TestsSearchResponse(BaseModel):
    items: List[TestsSearchItem]


# ------------------- 评审 -------------------


class TestsScope(BaseModel):
    source_ids: Optional[List[str]] = None
    path_prefix: Optional[str] = None


class CoverageConfig(BaseModel):
    top_k: int = 50
    threshold_cover: float = 0.82
    threshold_maybe: float = 0.75
    enable_llm_verifier: bool = False
    max_verify_per_scenario: int = 5


class QualityConfig(BaseModel):
    enable_quality_review: bool = True
    enable_llm_quality_review: bool = False


class ReviewRunRequest(BaseModel):
    requirements_scope: RequirementsScope
    tests_scope: TestsScope
    coverage: Optional[CoverageConfig] = None
    quality: Optional[QualityConfig] = None


class ReviewRun(BaseModel):
    run_id: str
    status: Literal["running", "done", "failed"]
    created_at: str
    config: Dict[str, Any] = Field(default_factory=dict)


class ReviewRunListResponse(BaseModel):
    items: List[ReviewRun]
    page: int
    page_size: int
    total: int


class ModuleCoverage(BaseModel):
    path: str
    total: int
    covered: int
    coverage_rate: float


class DiversityCount(BaseModel):
    type: Literal["normal", "exception", "boundary", "security"]
    count: int


class CoverageSummary(BaseModel):
    total_criteria: int
    covered_criteria: int
    coverage_rate: float
    module_breakdown: List[ModuleCoverage] = Field(default_factory=list)
    diversity_breakdown: List[DiversityCount] = Field(default_factory=list)


class IssueCount(BaseModel):
    type: str
    count: int


class QualitySummary(BaseModel):
    avg_completeness: float
    avg_consistency: float
    avg_executable: float
    issues_breakdown: List[IssueCount] = Field(default_factory=list)


class ReviewSummaryResponse(BaseModel):
    coverage: CoverageSummary
    quality: QualitySummary


class GapsRequest(BaseModel):
    priority: List[Literal["P0", "P1", "P2"]] = Field(default_factory=lambda: ["P0", "P1", "P2"])
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


class GapItem(BaseModel):
    priority: Literal["P0", "P1", "P2"]
    criterion: RequirementCriterion
    suggested_queries: List[str] = Field(default_factory=list)


class GapsResponse(BaseModel):
    items: List[GapItem]
    page: int
    page_size: int
    total: int


LinkType = Literal["coverage", "trace", "suggested"]
LinkStatus = Literal["covered", "maybe", "rejected"]


class Link(BaseModel):
    link_id: str
    run_id: str
    scenario_id: str
    criterion_id: str
    link_type: LinkType
    status: LinkStatus
    score_vector: float
    verifier_used: bool
    verifier_reason: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class LinksResponse(BaseModel):
    items: List[Link]


class QualityIssue(BaseModel):
    type: str
    severity: Literal["low", "medium", "high"]
    message: str


class QualityReviewItem(BaseModel):
    scenario_id: str
    completeness_score: int = Field(ge=0, le=100)
    consistency_score: int = Field(ge=0, le=100)
    executable_score: int = Field(ge=0, le=100)
    issues: List[QualityIssue] = Field(default_factory=list)
    llm_used: bool
    llm_suggestions: Optional[Dict[str, Any]] = None


class QualityIssueFilters(BaseModel):
    min_executable_score: Optional[float] = None
    max_executable_score: Optional[float] = None
    issue_types: Optional[List[str]] = None


class QualityIssuesRequest(BaseModel):
    filters: Optional[QualityIssueFilters] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=200)


class QualityIssuesResponse(BaseModel):
    items: List[QualityReviewItem]
    page: int
    page_size: int
    total: int


class ExportRequest(BaseModel):
    format: Literal["md", "xlsx", "json"]
    include_sections: List[str] = Field(default_factory=lambda: ["coverage", "diversity", "gaps", "quality"])


class LinkConfirmRequest(BaseModel):
    link_id: str
    action: Literal["confirm", "reject"]
    comment: Optional[str] = None


class OkResponse(BaseModel):
    ok: bool = True


class LlmSettingsRequest(BaseModel):
    provider: Literal["openai"]
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_verifier: str = "gpt-4o-mini"
    model_quality: str = "gpt-4o-mini"
