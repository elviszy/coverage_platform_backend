"""
开发调试入口文件

使用方法:
    python run.py               # 启动开发服务器 (带热重载)
    python run.py --debug       # 调试模式 (支持 IDE 断点调试)
    python run.py --prod        # 生产模式启动 (无热重载)
    python run.py --port 8080   # 指定端口
    python run.py --host 0.0.0.0 # 指定监听地址
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn


def setup_logging(debug: bool = True):
    """配置日志"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="测试覆盖率评审平台后端服务")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址 (默认: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口 (默认: 8000)",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        help="生产模式 (禁用热重载和自动调试)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式 (禁用热重载, 支持 IDE 断点调试)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数 (仅生产模式, 默认: 1)",
    )

    args = parser.parse_args()

    # 配置日志
    setup_logging(debug=not args.prod)

    if args.prod:
        # 生产模式
        logging.info("🚀 启动生产模式服务器")
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
        )
    elif args.debug:
        # 调试模式 (支持 IDE 断点)
        from app.main import app
        logging.info("🐛 启动调试模式服务器 (支持断点调试)")
        logging.info(f"📍 服务地址: http://{args.host}:{args.port}")
        logging.info(f"📚 API文档: http://{args.host}:{args.port}/docs")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="debug",
        )
    else:
        # 开发模式 (带热重载)
        logging.info("🔧 启动开发模式服务器 (热重载已启用)")
        logging.info(f"📍 服务地址: http://{args.host}:{args.port}")
        logging.info(f"📚 API文档: http://{args.host}:{args.port}/docs")
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["app"],
            log_level="debug",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n👋 服务已停止")
        sys.exit(0)
    except Exception as e:
        logging.error(f"❌ 启动失败: {e}")
        sys.exit(1)
