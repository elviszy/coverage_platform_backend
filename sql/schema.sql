-- 测试覆盖率评审平台（PostgreSQL + pgvector）建表脚本
-- 默认 embedding 维度为 1536，如需修改请同步调整 vector(1536)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS coverage_platform;
SET search_path TO coverage_platform, public;

-- 需求页面元数据（用于增量与追溯）
CREATE TABLE IF NOT EXISTS requirements_pages (
  page_id            TEXT PRIMARY KEY,
  page_url           TEXT NOT NULL,
  title              TEXT NOT NULL,
  version            INTEGER NOT NULL,
  body_storage       TEXT NOT NULL DEFAULT '',
  path               TEXT NOT NULL DEFAULT '',
  labels             JSONB NOT NULL DEFAULT '{}'::jsonb,
  fetched_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 兼容已建表场景：补充新增列
ALTER TABLE IF EXISTS requirements_pages
  ADD COLUMN IF NOT EXISTS body_storage TEXT NOT NULL DEFAULT '';

-- 页面附件（用于图片/文件的下载与追溯）
CREATE TABLE IF NOT EXISTS requirements_attachments (
  page_id           TEXT NOT NULL REFERENCES requirements_pages(page_id) ON DELETE CASCADE,
  attachment_id     TEXT NOT NULL,
  filename          TEXT NOT NULL,
  media_type        TEXT,
  file_path         TEXT NOT NULL,
  download_url      TEXT NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (page_id, attachment_id)
);

CREATE INDEX IF NOT EXISTS idx_requirements_attachments_page ON requirements_attachments(page_id);
CREATE INDEX IF NOT EXISTS idx_requirements_attachments_filename ON requirements_attachments(filename);

CREATE INDEX IF NOT EXISTS idx_requirements_pages_path ON requirements_pages(path);

-- 需求库：验收标准（表格行级）
CREATE TABLE IF NOT EXISTS requirements_criteria (
  criterion_id       TEXT PRIMARY KEY,
  page_id            TEXT NOT NULL REFERENCES requirements_pages(page_id) ON DELETE CASCADE,
  page_version       INTEGER NOT NULL,
  page_url           TEXT NOT NULL,
  title              TEXT NOT NULL,
  path               TEXT NOT NULL DEFAULT '',
  table_idx          INTEGER NOT NULL,
  row_idx            INTEGER NOT NULL,
  table_title        TEXT,
  headers            JSONB NOT NULL DEFAULT '[]'::jsonb,
  row_data           JSONB NOT NULL DEFAULT '{}'::jsonb,
  normalized_text    TEXT NOT NULL,
  embedding          VECTOR(1536) NOT NULL,
  is_active          BOOLEAN NOT NULL DEFAULT TRUE,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_requirements_criteria_page ON requirements_criteria(page_id);
CREATE INDEX IF NOT EXISTS idx_requirements_criteria_path ON requirements_criteria(path);
CREATE INDEX IF NOT EXISTS idx_requirements_criteria_active ON requirements_criteria(is_active);

DO $$
BEGIN
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_requirements_criteria_embedding_hnsw ON requirements_criteria USING hnsw (embedding vector_cosine_ops)';
EXCEPTION
  WHEN OTHERS THEN
    RAISE NOTICE 'skip idx_requirements_criteria_embedding_hnsw: %', SQLERRM;
END
$$;

-- 用例导入源（XMind）
CREATE TABLE IF NOT EXISTS tests_sources (
  source_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type        TEXT NOT NULL DEFAULT 'xmind',
  file_name          TEXT NOT NULL,
  file_hash          TEXT NOT NULL,
  imported_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_tests_sources_hash ON tests_sources(file_hash);

-- 用例库：测试场景
CREATE TABLE IF NOT EXISTS tests_scenarios (
  scenario_id        TEXT PRIMARY KEY,
  source_id          UUID NOT NULL REFERENCES tests_sources(source_id) ON DELETE CASCADE,
  title              TEXT NOT NULL,
  path               TEXT NOT NULL DEFAULT '',
  notes              TEXT,
  context_text       TEXT NOT NULL,
  embedding          VECTOR(1536) NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tests_scenarios_source ON tests_scenarios(source_id);
CREATE INDEX IF NOT EXISTS idx_tests_scenarios_path ON tests_scenarios(path);

DO $$
BEGIN
  EXECUTE 'CREATE INDEX IF NOT EXISTS idx_tests_scenarios_embedding_hnsw ON tests_scenarios USING hnsw (embedding vector_cosine_ops)';
EXCEPTION
  WHEN OTHERS THEN
    RAISE NOTICE 'skip idx_tests_scenarios_embedding_hnsw: %', SQLERRM;
END
$$;

-- 评审 Run
CREATE TABLE IF NOT EXISTS review_runs (
  run_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  status             TEXT NOT NULL DEFAULT 'running',
  requirements_scope JSONB NOT NULL DEFAULT '{}'::jsonb,
  tests_scope        JSONB NOT NULL DEFAULT '{}'::jsonb,
  config             JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_runs_created_at ON review_runs(created_at DESC);

-- 关联关系（需求 ↔ 场景）
CREATE TABLE IF NOT EXISTS kb_links (
  link_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id             UUID REFERENCES review_runs(run_id) ON DELETE SET NULL,
  scenario_id        TEXT NOT NULL REFERENCES tests_scenarios(scenario_id) ON DELETE CASCADE,
  criterion_id       TEXT NOT NULL REFERENCES requirements_criteria(criterion_id) ON DELETE CASCADE,
  link_type          TEXT NOT NULL DEFAULT 'coverage',
  status             TEXT NOT NULL DEFAULT 'maybe',
  score_vector       DOUBLE PRECISION NOT NULL,
  verifier_used      BOOLEAN NOT NULL DEFAULT FALSE,
  verifier_reason    TEXT,
  evidence           JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_links_run ON kb_links(run_id);
CREATE INDEX IF NOT EXISTS idx_kb_links_scenario ON kb_links(scenario_id);
CREATE INDEX IF NOT EXISTS idx_kb_links_criterion ON kb_links(criterion_id);
CREATE INDEX IF NOT EXISTS idx_kb_links_status ON kb_links(status);

CREATE UNIQUE INDEX IF NOT EXISTS uq_kb_links_run_pair
ON kb_links(run_id, scenario_id, criterion_id);

-- 覆盖度摘要（/reviews/runs/{id}/summary 快速读取）
CREATE TABLE IF NOT EXISTS review_summary (
  run_id             UUID PRIMARY KEY REFERENCES review_runs(run_id) ON DELETE CASCADE,
  total_criteria     INTEGER NOT NULL,
  covered_criteria   INTEGER NOT NULL,
  coverage_rate      DOUBLE PRECISION NOT NULL,
  module_breakdown   JSONB NOT NULL DEFAULT '[]'::jsonb,
  diversity_breakdown JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 质量评审（按场景粒度）
CREATE TABLE IF NOT EXISTS quality_review_items (
  run_id             UUID NOT NULL REFERENCES review_runs(run_id) ON DELETE CASCADE,
  scenario_id        TEXT NOT NULL REFERENCES tests_scenarios(scenario_id) ON DELETE CASCADE,
  completeness_score INTEGER NOT NULL,
  consistency_score  INTEGER NOT NULL,
  executable_score   INTEGER NOT NULL,
  issues             JSONB NOT NULL DEFAULT '[]'::jsonb,
  llm_used           BOOLEAN NOT NULL DEFAULT FALSE,
  llm_suggestions    JSONB,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, scenario_id)
);

CREATE INDEX IF NOT EXISTS idx_quality_review_items_run ON quality_review_items(run_id);
CREATE INDEX IF NOT EXISTS idx_quality_review_items_exec ON quality_review_items(executable_score);

-- 任务系统（可选，建议用于 /jobs/{job_id}）
CREATE TABLE IF NOT EXISTS jobs (
  job_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type               TEXT NOT NULL,
  status             TEXT NOT NULL,
  progress           DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  result             JSONB NOT NULL DEFAULT '{}'::jsonb,
  error              JSONB
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
