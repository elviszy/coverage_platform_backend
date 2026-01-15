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
    use_llm_refinement: bool = Field(default=False, description="是否使用 LLM 精确定位需求点")


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
    feature_points: str = ""  # LLM 提取的需求点/功能点（Markdown 格式）


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


# ==================== 公共测试标准管理 ====================


class PublicCriterionBase(BaseModel):
    """公共测试标准基础字段"""
    category: str = Field(description="测试类型（增删改、审核、查询、校验、导入、数值等）")
    test_point: str = Field(description="测试点名称")
    test_content: Optional[str] = Field(default=None, description="测试内容描述")


class PublicCriterionCreate(PublicCriterionBase):
    """创建公共测试标准请求"""
    pass


class PublicCriterionUpdate(BaseModel):
    """更新公共测试标准请求"""
    category: Optional[str] = None
    test_point: Optional[str] = None
    test_content: Optional[str] = None
    is_active: Optional[bool] = None


class PublicCriterion(PublicCriterionBase):
    """公共测试标准响应"""
    criterion_id: str
    keywords: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: str
    updated_at: str


class PublicCriteriaListResponse(BaseModel):
    """公共测试标准列表响应"""
    items: List[PublicCriterion]
    total: int


class PublicCriteriaImportRequest(BaseModel):
    """从 Markdown 导入公共测试标准"""
    content: str = Field(description="Markdown 文件内容（表格格式）")
    replace_all: bool = Field(default=False, description="是否替换全部现有数据")


class PublicCriteriaImportResponse(BaseModel):
    """导入结果响应"""
    imported: int = Field(description="成功导入的条数")
    updated: int = Field(description="更新的条数")
    skipped: int = Field(description="跳过的条数")
    errors: List[str] = Field(default_factory=list, description="错误信息列表")


class PublicCriteriaIndexResponse(BaseModel):
    """索引重建响应"""
    indexed: int = Field(description="已索引条数")
    failed: int = Field(description="失败条数")


# ==================== 覆盖度分析 ====================


class LLMEnhanceConfig(BaseModel):
    """LLM 增强配置"""
    enable_boundary_verify: bool = Field(default=True, description="启用边界匹配 LLM 验证（0.65~0.80 分数段）")
    enable_miss_suggestion: bool = Field(default=True, description="启用未覆盖项智能建议")
    enable_requirements_context: bool = Field(default=True, description="启用需求文档上下文增强")
    max_verify_count: int = Field(default=20, ge=1, le=100, description="每次分析最多 LLM 验证次数")
    max_suggestion_count: int = Field(default=10, ge=1, le=50, description="最多生成建议的未覆盖项数量")


class CoverageAnalysisConfigSchema(BaseModel):
    """覆盖度分析配置"""
    threshold_cover: float = Field(default=0.80, ge=0.5, le=1.0, description="覆盖阈值")
    threshold_partial: float = Field(default=0.65, ge=0.3, le=1.0, description="部分覆盖阈值（触发 LLM 验证）")
    enable_dynamic_threshold: bool = Field(default=True, description="启用多级阈值（根据文本长度动态调整）")
    embedding_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Embedding 相似度权重")
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="关键词命中权重")
    categories: Optional[List[str]] = Field(default=None, description="限定分析的类型（为空则分析全部）")
    llm: Optional[LLMEnhanceConfig] = Field(default=None, description="LLM 增强配置（为空则不启用 LLM）")


class CoverageAnalyzeRequest(BaseModel):
    """发起覆盖度分析请求"""
    xmind_source_id: str = Field(description="XMind 来源 ID")
    requirements_page_ids: Optional[List[str]] = Field(default=None, description="关联的需求页面 ID 列表（可选）")
    config: Optional[CoverageAnalysisConfigSchema] = Field(default=None, description="分析配置")


class MatchedScenario(BaseModel):
    """匹配的测试场景"""
    scenario_id: str
    title: str
    path: str
    score: float
    matched_keywords: List[str] = Field(default_factory=list)


class MatchedRequirement(BaseModel):
    """匹配的需求点"""
    page_id: str
    page_title: str
    text: str
    score: float


class CoverageResultItem(BaseModel):
    """单个公共标准的覆盖结果"""
    criterion_id: str
    category: str
    test_point: str
    test_content: Optional[str]
    status: Literal["covered", "partial", "missed"]
    best_score: float
    matched_keywords: List[str] = Field(default_factory=list)
    matched_scenarios: List[MatchedScenario] = Field(default_factory=list)
    matched_requirements: List[MatchedRequirement] = Field(default_factory=list)
    llm_verified: bool = False
    llm_reason: Optional[str] = None
    llm_suggestion: Optional[str] = None


class CategoryCoverageStats(BaseModel):
    """按类型分组的覆盖统计"""
    category: str
    total: int
    covered: int
    partial: int
    missed: int
    coverage_rate: float = Field(description="覆盖率百分比，如 72.5")


class CoverageAnalysisSummary(BaseModel):
    """覆盖度分析汇总"""
    total_criteria: int
    covered: int
    partial: int
    missed: int
    coverage_rate: float = Field(description="总体覆盖率百分比")
    by_category: List[CategoryCoverageStats] = Field(default_factory=list, description="按类型分组统计")
    requirements_linked: int = Field(default=0, description="有需求关联的测试点数量")
    llm_verified_count: int = Field(default=0, description="LLM 参与验证的数量")
    llm_suggestion_count: int = Field(default=0, description="生成了建议的未覆盖项数量")


class CoverageAnalysisResponse(BaseModel):
    """覆盖度分析结果响应"""
    run_id: str
    status: Literal["pending", "running", "completed", "failed"]
    xmind_source_id: str
    xmind_source_name: str
    requirements_pages: List[Dict[str, Any]] = Field(default_factory=list, description="关联的需求页面信息")
    config: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[CoverageAnalysisSummary] = None
    covered_items: List[CoverageResultItem] = Field(default_factory=list)
    partial_items: List[CoverageResultItem] = Field(default_factory=list)
    missed_items: List[CoverageResultItem] = Field(default_factory=list)
    created_at: str
    finished_at: Optional[str] = None


class CoverageRunListItem(BaseModel):
    """覆盖度分析历史列表项"""
    run_id: str
    xmind_source_id: str
    xmind_source_name: str
    status: str
    coverage_rate: Optional[float] = None
    total_criteria: Optional[int] = None
    created_at: str
    finished_at: Optional[str] = None


class CoverageRunListResponse(BaseModel):
    """覆盖度分析历史列表响应"""
    items: List[CoverageRunListItem]
    total: int

