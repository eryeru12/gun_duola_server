# Gunicorn Gevent Flask 项目

这是一个使用Gunicorn和Gevent的Flask项目模板。

## 功能特性
- 使用Gevent异步工作模式
- 预配置Gunicorn生产环境设置
- 简单的Flask示例应用

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 启动服务：
   ```bash
   ./run.sh
   ```

## 配置
- 服务端口：8000
- Worker数量：1 (可在gunicorn_config.py中调整)
- Worker类型：gevent

## 项目结构
- `app.py` - 主应用文件
- `gunicorn_config.py` - Gunicorn配置文件
- `requirements.txt` - 依赖文件
- `run.sh` - 启动脚本
