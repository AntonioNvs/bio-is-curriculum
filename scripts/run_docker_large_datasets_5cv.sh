#!/usr/bin/env bash
# RoBERTa-base, 5-fold CV, modos raw/is/cl/is_cl, biois_discrete.
# Datasets: yelp_2013, agnews, medline.
#
# Uso:
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_large_datasets_5cv.sh [GPU_ID]
#
# Exemplo (GPU 7):
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_large_datasets_5cv.sh 7
#
# Variaveis opcionais:
#   DATASETS   lista separada por espaco (default: "yelp_2013 agnews medline")
#   CONFIG     YAML de experimento (default: experiments/large_datasets_roberta_base_5cv.yaml)
#   CPUS       CPUs do container (default: 16)
#   MEMORY     memoria do container (default: 32g)
set -euo pipefail

GPU_ID="${1:-7}"

DATASETS="${DATASETS:-agnews medline}"
CONFIG="${CONFIG:-experiments/large_datasets_roberta_base_5cv.yaml}"
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32g}"

IMAGE="${IMAGE:-bio-is-curriculum:latest}"
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTAINER_WORKDIR="/app"

total_datasets=$(echo "${DATASETS}" | wc -w)
idx=0

echo "============================================================"
echo "  Large datasets — RoBERTa-base 5cv (biois_discrete)"
echo "============================================================"
echo "  Image    : ${IMAGE}"
echo "  GPU      : device=${GPU_ID}"
echo "  Config   : ${CONFIG}"
echo "  Datasets : ${DATASETS}"
echo "  Host dir : ${HOST_PROJECT_DIR}"
echo "============================================================"

for dataset in ${DATASETS}; do
  idx=$((idx + 1))
  echo ""
  echo "------------------------------------------------------------"
  echo "  [${idx}/${total_datasets}] Dataset: ${dataset}"
  echo "------------------------------------------------------------"

  docker run --rm \
    --gpus "device=${GPU_ID}" \
    --cpus="${CPUS}" \
    --memory="${MEMORY}" \
    -e CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    -e PYTHONHASHSEED="42" \
    -e OMP_NUM_THREADS="${CPUS}" \
    -e MKL_NUM_THREADS="${CPUS}" \
    -v "${HOST_PROJECT_DIR}/datasets:${CONTAINER_WORKDIR}/datasets" \
    -v "${HOST_PROJECT_DIR}/results:${CONTAINER_WORKDIR}/results" \
    -w "${CONTAINER_WORKDIR}" \
    "${IMAGE}" \
    python run.py "${CONFIG}" --dataset "${dataset}" --fail-fast
done

echo ""
echo "============================================================"
echo "  Experimento concluido."
echo "  Resultados em: ${HOST_PROJECT_DIR}/results/"
echo "============================================================"
