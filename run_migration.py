"""执行数据库迁移：添加 feature_points 字段"""
import psycopg

conn = psycopg.connect('postgresql://postgres:postgres@localhost:5432/postgres')
cur = conn.cursor()

# 确保 schema 存在
cur.execute("CREATE SCHEMA IF NOT EXISTS coverage_platform")
conn.commit()

# 检查表是否存在
cur.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'coverage_platform' 
        AND table_name = 'requirements_criteria'
    )
""")
table_exists = cur.fetchone()[0]

if table_exists:
    # 添加 feature_points 字段
    cur.execute("""
        ALTER TABLE coverage_platform.requirements_criteria 
        ADD COLUMN IF NOT EXISTS feature_points JSONB NOT NULL DEFAULT '[]'::jsonb
    """)
    conn.commit()
    print("Migration completed successfully: feature_points column added")
else:
    print("Table requirements_criteria does not exist yet. Run schema.sql first.")

conn.close()
