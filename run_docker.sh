#!/usr/bin/env bash
# Executa os experimentos do biO-IS-Curriculum dentro de um container Docker.
#
# Uso:
#   ./run_docker.sh [GPU_ID] [DATASET] [FOLD]
#
# Exemplos:
#   ./run_docker.sh            # GPU 7, dataset webkb, fold 0
#   ./run_docker.sh 0          # GPU 0
#   ./run_docker.sh 3 reuters  # GPU 3, dataset reuters
#   ./run_docker.sh 3 reuters 2  # GPU 3, dataset reuters, fold 2
set -euo pipefail

# ── Configurações ──────────────────────────────────────────────────────────────
GPU_ID="${1:-7}"
DATASET="${2:-webkb}"
FOLD="${3:-0}"

IMAGE="my_image:latest"
HOST_DIR="/workdir/antonioneves"
CONTAINER_DIR="/workspace/antonioneves"
WORKDIR="${CONTAINER_DIR}/bio-is-curriculum"

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
echo "  Workdir : ${WORKDIR}"
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
    -v "${HOST_DIR}:${CONTAINER_DIR}" \
    -w "${WORKDIR}" \
    "${IMAGE}" \
    bash -c "${CMD}"

  echo "------------------------------------------------------------"
  echo "<<< concluido: ${CMD}"
done

echo ""
echo "============================================================"
echo "  Todos os experimentos concluidos."
echo "============================================================"
