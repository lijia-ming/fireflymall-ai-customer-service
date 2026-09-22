# ============================================================
# Dockerfile - 智能客服系统 (FastAPI + LangGraph)
# 基于 Python 3.12 slim 镜像，多阶段构建优化镜像体积
# ============================================================

# ---------- 构建阶段 ----------
FROM python:3.12-slim AS builder

WORKDIR /build

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt 以利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- 运行阶段 ----------
FROM python:3.12-slim AS runtime

WORKDIR /app

# 安装运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制已安装的 Python 包
COPY --from=builder /root/.local /root/.local

# 确保 pip 安装的 CLI 工具在 PATH 中
ENV PATH=/root/.local/bin:$PATH

# HuggingFace 环境变量：容器入口是 uvicorn 而非 run.py，
# run.py 里那段读取 config['huggingface'] 的逻辑不会执行，
# 所以 mirror / download_dir 必须在镜像层固化，否则 tokenizer 会下载到
# ~/.cache 且不走镜像站。与 config.yaml 的 huggingface 段保持一致；
# 需要海外直连时用 -e HF_ENDPOINT=https://huggingface.co 覆盖。
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/huggingface_cache

# 复制项目源代码（排除 .dockerignore 中定义的文件）
COPY SPO/ ./SPO/
COPY agent/ ./agent/
COPY routes/ ./routes/
COPY Tools/ ./Tools/
COPY load_config/ ./load_config/
COPY model/ ./model/
COPY main.py .

# 创建运行时目录（可由 volume 覆盖）
RUN mkdir -p /app/data/uploads /app/data/downloads /app/cache /app/database /app/log

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ai/health')" || exit 1

EXPOSE 8000

# 使用 uvicorn 启动（start_app 是工厂函数，需显式声明 --factory）
CMD ["uvicorn", "main:start_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
