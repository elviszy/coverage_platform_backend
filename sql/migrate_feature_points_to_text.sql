-- =====================================================
-- 数据库迁移脚本：feature_points 字段从 JSONB 改为 TEXT
-- =====================================================
-- 执行时间：2026-01-03
-- 说明：将 feature_points 字段从 JSONB 改为 TEXT，存储 Markdown 格式
-- 注意：此迁移会清空现有的 feature_points 数据
-- =====================================================

-- 先检查表是否存在
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'coverage_platform' 
        AND table_name = 'requirements_criteria'
    ) THEN
        -- 1. 删除旧的 JSONB 列（如果存在）
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = 'coverage_platform' 
            AND table_name = 'requirements_criteria' 
            AND column_name = 'feature_points'
        ) THEN
            ALTER TABLE coverage_platform.requirements_criteria DROP COLUMN feature_points;
            RAISE NOTICE '已删除旧的 feature_points 列';
        END IF;
        
        -- 2. 添加新的 TEXT 类型列
        ALTER TABLE coverage_platform.requirements_criteria 
        ADD COLUMN feature_points TEXT NOT NULL DEFAULT '';
        
        RAISE NOTICE '已添加新的 feature_points 列（TEXT 类型）';
        
        -- 3. 添加列注释
        COMMENT ON COLUMN coverage_platform.requirements_criteria.feature_points 
        IS 'LLM 提取的需求点/功能点（Markdown 格式）';
        
    ELSE
        RAISE NOTICE '表 coverage_platform.requirements_criteria 不存在，跳过迁移';
    END IF;
END $$;
