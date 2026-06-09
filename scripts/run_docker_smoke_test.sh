#!/usr/bin/env bash
# Smoke test multi-mode: roda 1 fold para cada token de MODES dentro de um único
# container Docker, agrupados em results/<experiment_id>/<mode>_fold<k>/.
# Útil para validar rápido o pipeline ponta-a-ponta antes de disparar o full CV.
#
# Uso:
#   IMAGE=<imagem> ./scripts/run_docker_smoke_test.sh [GPU_ID] [DATASET]
#
# Exemplos:
#   # Default: roda só 'is' no fold 0 do webkb
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_smoke_test.sh
#
#   # Compara raw (sem IS/CL), is, cl, is_cl e o baseline 1 da literatura
#   IMAGE=bio-is-curriculum:latest MODES="raw is cl is_cl b1" \
#     ./scripts/run_docker_smoke_test.sh 0 webkb
#
#   # Token "bN" (regex ^b[0-9]+$) é mapeado para --baseline N (ver BASELINES.md);
#   # qualquer outro token é tratado como --mode <token>.
#
# Variáveis de ambiente opcionais:
#   MODEL         lr | roberta                     (default: roberta)
#   MODES         lista separada por espaço         (default: "is")
#   FOLD          fold único usado para todos       (default: 0)
#   N_SPLITS      n splits do dataset               (default: 10)
#   EPOCHS        épocas de treino (raw/is)         (default: 6)
#   EPOCHS_PP     épocas por fase (cl/is_cl/bN)     (default: 2)
#   BATCH_SIZE    batch de treino                   (default: 32)
#   LR            learning rate                     (default: 2e-5)
#   WEIGHT_DECAY  L2 weight decay                   (default: 1e-3)
#   WARMUP_RATIO  fração de warmup linear           (default: 0.06)
#   BETA          beta do BIOIS                     (default: 0.3)
#   THETA         theta do BIOIS                    (default: 0.2)
#   CURRICULUM_METHOD metodo de curriculum (cl/is_cl/bN)
#                                                  (default: biois_discrete)
#   EXPERIMENT_ID força um experiment-id especifico (default: smoke-YYYYmmdd-HHMMSS)
#   CPUS          limite de CPUs do container       (default: 16)
#   MEMORY        limite de memória do container    (default: 32g)
#
# Os limites de CPU/MEMORY casam com run_docker_full_cv.sh para que o mesmo
# (fold, mode) produza métricas idênticas entre smoke e full — caso contrário
# diferenças de threading do BLAS/saga viram drift numérico.
set -euo pipefail

# ── Configurações ──────────────────────────────────────────────────────────────
GPU_ID="${1:-7}"
DATASET="${2:-reuters90}"
N_SPLITS="${N_SPLITS:-10}"

MODEL="${MODEL:-roberta}"
MODES="${MODES:-is_cl}"
FOLD="${FOLD:-0}"
EPOCHS="${EPOCHS:-6}"
EPOCHS_PP="${EPOCHS_PP:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
BETA="${BETA:-0.3}"
THETA="${THETA:-0.2}"
CURRICULUM_METHOD="${CURRICULUM_METHOD:-spcl_soft}"

EXPERIMENT_ID="${EXPERIMENT_ID:-smoke-$(date +%Y%m%d-%H%M%S)}"

CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32g}"

IMAGE="${IMAGE:-bio-is-curriculum:latest}"
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTAINER_WORKDIR="/app"

# ── Monta o comando interno: loop pelos MODES, agrupados em um experiment-id ──
# Para cada token: se for "bN" vira --baseline N, senao vira --mode <token>.
# Tudo continua mesmo se um modo falhar (||true) para ter o relatorio completo.
COMMON_ARGS=(
  "--fold ${FOLD}"
  "--n-splits ${N_SPLITS}"
  "--experiment-id ${EXPERIMENT_ID}"
  "--model ${MODEL}"
  "--beta ${BETA}"
  "--theta ${THETA}"
  "--curriculum-method ${CURRICULUM_METHOD}"
  "--epochs ${EPOCHS}"
  "--epochs-per-phase ${EPOCHS_PP}"
  "--batch-size ${BATCH_SIZE}"
  "--lr ${LR}"
  "--weight-decay ${WEIGHT_DECAY}"
  "--warmup-ratio ${WARMUP_RATIO}"
  "--data_dir datasets"
  "--results-dir results"
)
COMMON_STR="${COMMON_ARGS[*]}"

INNER=""
for tok in ${MODES}; do
  if [[ "${tok}" =~ ^b[0-9]+$ ]]; then
    bln_n="${tok#b}"
    mode_flag="--baseline ${bln_n}"
    label="baseline ${bln_n}"
  else
    mode_flag="--mode ${tok}"
    label="mode ${tok}"
  fi
  INNER+="echo; echo '============================================================'; "
  INNER+="echo '>> Rodando ${label} (fold ${FOLD})'; "
  INNER+="echo '============================================================'; "
  INNER+="python main.py ${DATASET} ${mode_flag} ${COMMON_STR} || echo '!! ${label} falhou (continuando)'; "
done

# ── Resumo ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  SMOKE TEST multi-mode"
echo "============================================================"
echo "  Image          : ${IMAGE}"
echo "  GPU            : device=${GPU_ID}"
echo "  Dataset        : ${DATASET}  |  N-splits: ${N_SPLITS}"
echo "  Modes          : ${MODES}"
echo "  Fold           : ${FOLD}"
echo "  Model          : ${MODEL}"
echo "  Epochs (is/bs) : ${EPOCHS}   |  per-phase (cl/is_cl/bN): ${EPOCHS_PP}"
echo "  Batch          : ${BATCH_SIZE}"
echo "  LR             : ${LR}  |  WD: ${WEIGHT_DECAY}  |  Warmup: ${WARMUP_RATIO}"
echo "  Beta/Theta     : ${BETA}/${THETA}"
echo "  Curriculum     : ${CURRICULUM_METHOD}"
echo "  Experiment ID  : ${EXPERIMENT_ID}"
echo "  Host dir       : ${HOST_PROJECT_DIR}"
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
  bash -c "${INNER}"

echo ""
echo "============================================================"
echo "  Smoke test concluido."
echo "  Resultados : results/${EXPERIMENT_ID}/<mode>_fold${FOLD}/"
echo "============================================================"
