# 开发调试指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制环境变量示例文件：
```bash
cp .env.example .env
```

根据实际情况修改 `.env` 文件中的配置。

### 3. 启动服务

#### 方式一：使用 run.py (推荐)

```bash
# 开发模式 (带热重载)
python run.py

# 指定端口
python run.py --port 8080

# 指定监听地址
python run.py --host 0.0.0.0

# 生产模式
python run.py --prod --workers 4
```

#### 方式二：直接使用 uvicorn

```bash
# 开发模式
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. 访问服务

- API服务: http://127.0.0.1:8000
- API文档 (Swagger): http://127.0.0.1:8000/docs
- API文档 (ReDoc): http://127.0.0.1:8000/redoc

## IDE 调试配置

### VSCode

已提供 `.vscode/launch.json` 配置文件，包含以下调试配置：

1. **Python: FastAPI** - 直接启动 uvicorn
2. **Python: Run.py (开发模式)** - 使用 run.py 启动
3. **Python: 当前文件** - 调试当前打开的文件

使用方法：
1. 按 `F5` 或点击调试面板的"开始调试"
2. 选择对应的调试配置
3. 设置断点进行调试

### PyCharm

1. 打开 Run/Debug Configurations
2. 添加新的 Python 配置
3. 配置如下：
   - **Script path**: 选择 `run.py`
   - **Working directory**: 项目根目录
   - 或者：
     - **Module name**: `uvicorn`
     - **Parameters**: `app.main:app --reload --host 127.0.0.1 --port 8000`

## 常用调试技巧

### 1. 查看日志

开发模式下会输出详细的 DEBUG 级别日志，包括：
- 请求路径和参数
- SQL 查询
- 异常堆栈

### 2. 交互式 API 文档

访问 http://127.0.0.1:8000/docs 可以：
- 查看所有 API 接口
- 在线测试接口
- 查看请求/响应模型

### 3. 数据库调试

在代码中添加 SQL 日志输出：

```python
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

### 4. 热重载

开发模式下修改代码后会自动重载，无需手动重启服务。

监控目录：`app/`

### 5. 断点调试

在 IDE 中设置断点，使用调试模式启动即可进行断点调试。

## 数据库配置

确保 PostgreSQL 已安装并启动，然后创建数据库：

```sql
CREATE DATABASE coverage_platform;
```

使用 pgvector 扩展：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## 常见问题

### 端口被占用

修改启动端口：
```bash
python run.py --port 8080
```

### 数据库连接失败

检查 `.env` 文件中的 `DATABASE_URL` 配置是否正确。

### 模块导入错误

确保在项目根目录下执行命令，或设置 `PYTHONPATH`：
```bash
export PYTHONPATH=.
```
