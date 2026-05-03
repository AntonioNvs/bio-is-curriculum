#!/usr/bin/env bash
# Executa os experimentos do biO-IS-Curriculum dentro de um container Docker.
#
# Uso:
#   IMAGE=<imagem> ./run_docker.sh [GPU_ID] [DATASET] [FOLD]
#
# Exemplos:
#   IMAGE=meu_registry/minha_imagem:latest ./run_docker.sh
#   IMAGE=meu_registry/minha_imagem:latest ./run_docker.sh 0
#   IMAGE=meu_registry/minha_imagem:latest ./run_docker.sh 3 reuters
#   IMAGE=meu_registry/minha_imagem:latest ./run_docker.sh 3 reuters 2
set -euo pipefail

# ── Configurações ──────────────────────────────────────────────────────────────
GPU_ID="${1:-7}"
DATASET="${2:-webkb}"
FOLD="${3:-0}"

IMAGE="${IMAGE:-bio-is-curriculum:latest}"
# Diretório do projeto no HOST (onde estão datasets/ e results/)
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(pwd)}"
# WORKDIR dentro do container, definido no Dockerfile como /app
CONTAINER_WORKDIR="/app"
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32g}"

# ── Comandos a executar dentro do container ────────────────────────────────────
# Cada linha é um experimento independente (os 4 modos da matriz IS × CL).
COMMANDS=(
  "uv run python main.py ${DATASET} --data_dir datasets --fold ${FOLD} --mode baseline --epochs 6"
  "uv run python main.py ${DATASET} --data_dir datasets --fold ${FOLD} --mode is --epochs 6 --beta 0.3 --theta 0.2"
  "uv run python main.py ${DATASET} --data_dir datasets --fold ${FOLD} --mode cl --epochs-per-phase 2"
  "uv run python main.py ${DATASET} --data_dir datasets --fold ${FOLD} --mode is_cl --epochs-per-phase 2 --beta 0.3 --theta 0.2"
)

# ── Resumo ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Image   : ${IMAGE}"
echo "  GPU     : device=${GPU_ID}"
echo "  Dataset : ${DATASET}  |  Fold: ${FOLD}"
  echo "  Host dir: ${HOST_PROJECT_DIR}"
echo "============================================================"

# ── Execução ───────────────────────────────────────────────────────────────────
for CMD in "${COMMANDS[@]}"; do
  echo ""
  echo ">>> ${CMD}"
  echo "------------------------------------------------------------"

  docker run --rm \
    --gpus "device=${GPU_ID}" \
    --cpus="${CPUS}" \
    --memory="${MEMORY}" \
    -v "${HOST_PROJECT_DIR}/datasets:${CONTAINER_WORKDIR}/datasets" \
    -v "${HOST_PROJECT_DIR}/results:${CONTAINER_WORKDIR}/results" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE}" \
    bash -c "${CMD}"

  echo "------------------------------------------------------------"
  echo "<<< concluido: ${CMD}"
done

echo ""
echo "============================================================"
echo "  Todos os experimentos concluidos."
echo "============================================================"
