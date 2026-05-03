# Base com CUDA 12.1 + cuDNN 8 + Python 3.11 (compatível com torch>=2.2)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Instala o uv
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && curl -Ls https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Instala só as dependências externas (sem instalar o projeto como pacote)
COPY pyproject.toml README.md ./
RUN uv sync --no-install-project

# Copia o código
COPY main.py ./
COPY src/ ./src/

# Garante que os módulos em src/ sejam encontrados diretamente
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python", "main.py"]
