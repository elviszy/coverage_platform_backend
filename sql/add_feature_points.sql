-- 添加 feature_points 字段到 requirements_criteria 表
-- 用于存储 LLM 提取的功能点列表

ALTER TABLE coverage_platform.requirements_criteria 
ADD COLUMN IF NOT EXISTS feature_points JSONB NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN coverage_platform.requirements_criteria.feature_points IS 'LLM 提取的功能点列表，格式: [{"title": "...", "description": "...", "source_excerpt": "..."}]';
