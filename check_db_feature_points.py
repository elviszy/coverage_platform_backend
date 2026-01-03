import asyncio
import os
from sqlalchemy import text
from app.db import async_session_factory

async def check_feature_points():
    async with async_session_factory() as db:
        # 获取最近更新的一条记录
        result = await db.execute(text("""
            SELECT criterion_id, feature_points, updated_at 
            FROM coverage_platform.requirements_criteria 
            ORDER BY created_at DESC 
            LIMIT 1
        """))
        row = result.fetchone()
        
        if row:
            print(f"ID: {row[0]}")
            print(f"Updated At: {row[2]}")
            print("-" * 50)
            print("Feature Points Content:")
            print(row[1])
            print("-" * 50)
            
            if row[1] and row[1].strip().startswith(("-", "#", "*")):
                print("【格式检查】看起来是 Markdown 格式 ✓")
                if "模块" in row[1] or "功能点" in row[1]:
                    print("【内容检查】包含新版关键词（模块/功能点） ✓")
                else:
                    print("【内容检查】未发现新版关键词，可能是旧 Markdown 或规则提取结果 ?")
            elif row[1] and row[1].strip().startswith(("[", "{")):
                print("【格式检查】看起来是 JSON 格式 ✗ (未迁移成功?)")
            else:
                print("【格式检查】无法识别或为空")
        else:
            print("数据库中没有记录。")

if __name__ == "__main__":
    asyncio.run(check_feature_points())
