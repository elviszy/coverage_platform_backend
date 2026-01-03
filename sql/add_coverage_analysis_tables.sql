-- 公共测试用例覆盖度分析 - 增量迁移脚本
-- 运行方式: psql -U <user> -d <database> -f add_coverage_analysis_tables.sql
-- 或在 pgAdmin 等工具中执行

SET search_path TO coverage_platform, public;

-- 开始事务
BEGIN;

-- 公共测试标准（知识库）
CREATE TABLE IF NOT EXISTS public_test_criteria (
  criterion_id       TEXT PRIMARY KEY,
  category           VARCHAR(100) NOT NULL,
  test_point         VARCHAR(500) NOT NULL,
  test_content       TEXT,
  normalized_text    TEXT NOT NULL,
  keywords           JSONB NOT NULL DEFAULT '[]'::jsonb,
  embedding          VECTOR(1536),
  is_active          BOOLEAN NOT NULL DEFAULT TRUE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public_test_criteria IS '公共测试标准知识库，存储通用测试点';
COMMENT ON COLUMN public_test_criteria.category IS '测试类型：增删改、审核、查询、校验、导入、数值等';
COMMENT ON COLUMN public_test_criteria.test_point IS '测试点名称';
COMMENT ON COLUMN public_test_criteria.test_content IS '测试内容详细描述';
COMMENT ON COLUMN public_test_criteria.keywords IS '关键词列表，用于双重校验匹配';
COMMENT ON COLUMN public_test_criteria.embedding IS '向量嵌入，用于语义匹配';

CREATE INDEX IF NOT EXISTS idx_public_test_criteria_category ON public_test_criteria(category);
CREATE INDEX IF NOT EXISTS idx_public_test_criteria_active ON public_test_criteria(is_active);

-- 向量索引（如果失败则跳过）
DO $$
BEGIN
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_public_test_criteria_embedding_hnsw ON public_test_criteria USING hnsw (embedding vector_cosine_ops)';
EXCEPTION
  WHEN OTHERS THEN
    RAISE NOTICE 'skip idx_public_test_criteria_embedding_hnsw: %', SQLERRM;
END
$$;


-- 覆盖度分析任务
CREATE TABLE IF NOT EXISTS coverage_analysis_runs (
  run_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  xmind_source_id    UUID NOT NULL REFERENCES tests_sources(source_id) ON DELETE CASCADE,
  requirements_page_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
  status             VARCHAR(50) NOT NULL DEFAULT 'pending',
  config             JSONB NOT NULL DEFAULT '{}'::jsonb,
  summary            JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at        TIMESTAMPTZ
);

COMMENT ON TABLE coverage_analysis_runs IS '覆盖度分析任务记录';
COMMENT ON COLUMN coverage_analysis_runs.requirements_page_ids IS '关联的需求页面 ID 列表';
COMMENT ON COLUMN coverage_analysis_runs.status IS '任务状态：pending/running/completed/failed';
COMMENT ON COLUMN coverage_analysis_runs.config IS '分析配置（阈值、LLM 配置等）';
COMMENT ON COLUMN coverage_analysis_runs.summary IS '汇总结果（覆盖率、按类型统计等）';

CREATE INDEX IF NOT EXISTS idx_coverage_analysis_runs_source ON coverage_analysis_runs(xmind_source_id);
CREATE INDEX IF NOT EXISTS idx_coverage_analysis_runs_status ON coverage_analysis_runs(status);
CREATE INDEX IF NOT EXISTS idx_coverage_analysis_runs_created_at ON coverage_analysis_runs(created_at DESC);


-- 覆盖度分析结果详情
CREATE TABLE IF NOT EXISTS coverage_analysis_results (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id             UUID NOT NULL REFERENCES coverage_analysis_runs(run_id) ON DELETE CASCADE,
  criterion_id       TEXT NOT NULL REFERENCES public_test_criteria(criterion_id) ON DELETE CASCADE,
  status             VARCHAR(50) NOT NULL,
  best_score         DOUBLE PRECISION NOT NULL DEFAULT 0,
  matched_keywords   JSONB NOT NULL DEFAULT '[]'::jsonb,
  matched_scenarios  JSONB NOT NULL DEFAULT '[]'::jsonb,
  matched_requirements JSONB NOT NULL DEFAULT '[]'::jsonb,
  llm_verified       BOOLEAN NOT NULL DEFAULT FALSE,
  llm_reason         TEXT,
  llm_suggestion     TEXT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE coverage_analysis_results IS '覆盖度分析结果详情，每个公共标准对应一条记录';
COMMENT ON COLUMN coverage_analysis_results.status IS '覆盖状态：covered/partial/missed';
COMMENT ON COLUMN coverage_analysis_results.best_score IS '最高匹配分数（Embedding*0.7 + 关键词*0.3）';
COMMENT ON COLUMN coverage_analysis_results.matched_keywords IS '匹配的关键词列表';
COMMENT ON COLUMN coverage_analysis_results.matched_scenarios IS '匹配的测试场景列表';
COMMENT ON COLUMN coverage_analysis_results.matched_requirements IS '关联的需求点列表';
COMMENT ON COLUMN coverage_analysis_results.llm_verified IS '是否经过 LLM 二次验证';
COMMENT ON COLUMN coverage_analysis_results.llm_suggestion IS 'LLM 生成的补充建议';

CREATE INDEX IF NOT EXISTS idx_coverage_analysis_results_run ON coverage_analysis_results(run_id);
CREATE INDEX IF NOT EXISTS idx_coverage_analysis_results_criterion ON coverage_analysis_results(criterion_id);
CREATE INDEX IF NOT EXISTS idx_coverage_analysis_results_status ON coverage_analysis_results(status);

-- 提交事务
COMMIT;

-- 输出结果
SELECT 'Migration completed successfully!' AS result;
SELECT 
  (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'coverage_platform' AND table_name = 'public_test_criteria') AS public_test_criteria_exists,
  (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'coverage_platform' AND table_name = 'coverage_analysis_runs') AS coverage_analysis_runs_exists,
  (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'coverage_platform' AND table_name = 'coverage_analysis_results') AS coverage_analysis_results_exists;
