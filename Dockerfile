# Base com CUDA 12.1 + cuDNN 8 + Python 3.11 (compatível com torch>=2.2)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Evita prompts interativos do apt
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema e o uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && curl -Ls https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copia somente os arquivos de definição de dependências primeiro
# para aproveitar o cache do Docker ao reinstalar
COPY pyproject.toml ./
COPY README.md ./

# Instala as dependências (sem o torch — já vem na imagem base com CUDA)
# O flag --no-build-isolation garante compatibilidade com o torch pré-instalado
RUN uv sync --no-build-isolation

# Copia o resto do código
COPY main.py ./
COPY src/ ./src/

# Variável para que o torch encontre a GPU corretamente
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run", "python", "src/cli.py"]
