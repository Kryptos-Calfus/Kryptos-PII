# The hosted PII extension runtime.
#
# The fine-tuned checkpoint is ~850 MB and is deliberately NOT baked into this
# image. Baking it would make every code change an 850 MB push and would tie the
# model's release cycle to the service's. It is mounted at runtime instead and
# located with KRYPTOS_PII_MODEL_DIR; the Helm chart mounts a volume there.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # No weights are fetched at runtime: the checkpoint is mounted, and a
    # surprise download inside a security component is not acceptable.
    HF_HUB_OFFLINE=1 \
    KRYPTOS_PII_MODEL_DIR=/models/laya-pii

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "fastapi>=0.115" "uvicorn[standard]>=0.32" "pydantic>=2.9" "pyyaml>=6" \
      "laya==0.3.9" "torch>=2.4" "safetensors>=0.4"

COPY kryptos_pii/ ./kryptos_pii/
COPY finetune/common.py ./finetune/common.py
COPY extension.yaml ./extension.yaml
RUN touch ./finetune/__init__.py

# Nothing here needs to write, so nothing here may.
RUN useradd --create-home --uid 10001 kryptos
USER kryptos

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "kryptos_pii.service:app", "--host", "0.0.0.0", "--port", "8080"]
