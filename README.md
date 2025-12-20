# 测试覆盖率评审平台后端（MVP）

## 目录说明
- `app/`：FastAPI 应用
- `sql/schema.sql`：PostgreSQL + pgvector 建表脚本（默认 embedding 维度 1536）
- `requirements.txt`：Python 依赖

## 快速启动（开发）

1) 安装依赖

```bash
pip install -r coverage_platform_backend/requirements.txt
```

2) 初始化数据库

- 创建数据库：`coverage_platform`
- 执行建表脚本：`coverage_platform_backend/sql/schema.sql`

3) 启动服务

```bash
uvicorn app.main:app --reload
```

默认文档地址：
- Swagger UI：`http://127.0.0.1:8000/docs`

## 配置
主要通过环境变量配置（MVP）：
- `DATABASE_URL`：数据库连接串（SQLAlchemy 格式）
- `OPENAI_API_KEY`：OpenAI Key
- `OPENAI_BASE_URL`：可选
- `EMBEDDING_DIM`：向量维度（默认 1536；更改维度需要同步调整数据库字段类型）
- `CONFLUENCE_BASE_URL`：Confluence Data Center 地址（可不带协议，后端会自动补全 `https://`）
- `CONFLUENCE_TOKEN`：Confluence Token（Bearer）

如果你使用本仓库根目录的 `docker-compose.yml` 启动 PostgreSQL（端口映射 `5432:5432`），则可使用：

```bash
set DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/coverage_platform
```

## 前端工程（Vue3）

前端目录：`coverage_platform_frontend/`

1) 安装依赖

```bash
npm install
```

2) 启动前端

```bash
npm run dev
```

默认地址：
- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`

说明：前端已在 `vite.config.ts` 配置代理，将 `/api/*` 转发到后端（默认 `127.0.0.1:8000`）。

## 前后端联调要点

- 任务查询：后端接口统一返回 `job_id`，可通过 `GET /api/v1/jobs/{job_id}` 查看进度、结果与错误。
- 递归导入 Confluence：`POST /api/v1/requirements/import/confluence` 支持 `recursive/max_depth/include_attachments`。
- 附件入库：递归导入若开启 `include_attachments=true`，会下载附件到本地并写入 `requirements_attachments`。

## 数据库初始化注意

- 本项目 schema 可能随功能迭代新增表或字段（例如 `requirements_pages.body_storage`、`requirements_attachments`）。
- 如你是从旧版本升级，请重新执行 `coverage_platform_backend/sql/schema.sql`（脚本内使用了 `IF NOT EXISTS` 与 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 进行兼容）。
