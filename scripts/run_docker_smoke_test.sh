#!/usr/bin/env bash
# Executa um único experimento (fold 0, modo baseline) dentro de um container
# Docker para verificar rapidamente se as métricas estão sendo calculadas e
# registradas corretamente, sem a sobrecarga de um CV completo.
#
# Uso:
#   IMAGE=<imagem> ./scripts/run_docker_smoke_test.sh [GPU_ID] [DATASET]
#
# Exemplos:
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_smoke_test.sh
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_smoke_test.sh 0 webkb
#
# Variáveis de ambiente opcionais:
#   MODEL      modelo a usar: lr | roberta    (default: roberta)
#   MODE       modo de execução               (default: baseline)
#   FOLD       fold a usar                    (default: 0)
#   N_SPLITS   número de splits do dataset    (default: 10)
#   EPOCHS     épocas de treino (baseline/is)  (default: 6)
#   EPOCHS_PP  épocas por fase (cl/is_cl)      (default: 2)
#   LR         learning rate                   (default: 2e-5)
#   WEIGHT_DECAY  L2 weight decay              (default: 1e-3)
#   WARMUP_RATIO  fração de warmup linear      (default: 0.06)
#   BETA       beta do BIOIS                   (default: 0.3)
#   THETA      theta do BIOIS                  (default: 0.2)
#   CPUS       limite de CPUs do container     (default: 16)
#   MEMORY     limite de memória do container  (default: 32g)
#
# Os limites de CPU/MEMORY casam com run_docker_full_cv.sh para garantir que
# (smoke, full) produzam métricas idênticas no mesmo (fold, mode) — qualquer
# diferença de threading do BLAS/saga vira fonte de drift numérico.
set -euo pipefail

# ── Configurações ──────────────────────────────────────────────────────────────
GPU_ID="${1:-7}"
DATASET="${2:-webkb}"
N_SPLITS="${N_SPLITS:-10}"

MODEL="${MODEL:-roberta}"
MODE="${MODE:-is}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-6}"
EPOCHS_PP="${EPOCHS_PP:-2}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
BETA="${BETA:-0.3}"
THETA="${THETA:-0.2}"

# Mesma envelope de recursos do full_cv para evitar drift de threads/BLAS.
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32g}"

IMAGE="${IMAGE:-bio-is-curriculum:latest}"
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTAINER_WORKDIR="/app"

# ── Monta o comando main.py ────────────────────────────────────────────────────
CMD="python main.py ${DATASET}"
CMD="${CMD} --mode ${MODE}"
CMD="${CMD} --fold ${FOLD}"
CMD="${CMD} --n-splits ${N_SPLITS}"
CMD="${CMD} --model ${MODEL}"
CMD="${CMD} --beta ${BETA}"
CMD="${CMD} --theta ${THETA}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --epochs-per-phase ${EPOCHS_PP}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --weight-decay ${WEIGHT_DECAY}"
CMD="${CMD} --warmup-ratio ${WARMUP_RATIO}"
CMD="${CMD} --data_dir datasets"
CMD="${CMD} --results-dir results"

# ── Resumo ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  SMOKE TEST — verificacao de metricas (execucao unica)"
echo "============================================================"
echo "  Image    : ${IMAGE}"
echo "  GPU      : device=${GPU_ID}"
echo "  Dataset  : ${DATASET}  |  N-splits: ${N_SPLITS}"
echo "  Mode     : ${MODE}  |  Fold: ${FOLD}"
echo "  Model    : ${MODEL}  |  Epochs: ${EPOCHS}"
echo "  LR       : ${LR}  |  WD: ${WEIGHT_DECAY}  |  Warmup: ${WARMUP_RATIO}"
echo "  Host dir : ${HOST_PROJECT_DIR}"
echo "  Comando  : ${CMD}"
echo "============================================================"

# ── Execução ───────────────────────────────────────────────────────────────────
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
  bash -c "${CMD}"

echo ""
echo "============================================================"
echo "  Smoke test concluido. Verifique as metricas acima."
echo "  Resultados salvos em: results/<run_id>/"
echo "============================================================"
