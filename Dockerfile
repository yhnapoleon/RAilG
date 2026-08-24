# RAilG 应用镜像。OpenSearch 由 docker-compose 单独起。
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖:pyproject 不变时这层走缓存
COPY pyproject.toml README.md ./
COPY railg/__init__.py railg/__init__.py
RUN pip install --no-cache-dir -e . && rm -rf /root/.cache

COPY railg/ railg/
COPY web/ web/
COPY config.yaml ./

# 会话库和 URL 缓存都落在这里,compose 里挂成卷
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "railg.api:app", "--host", "0.0.0.0", "--port", "8000"]
