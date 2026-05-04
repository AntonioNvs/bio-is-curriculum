#!/usr/bin/env bash
# Executa os experimentos multi-fold do biO-IS-Curriculum dentro de um container Docker.
#
# Uso:
#   IMAGE=<imagem> ./run_docker.sh [GPU_ID] [DATASET] [N_SPLITS]
#
# Exemplos:
#   IMAGE=bio-is-curriculum:latest ./run_docker.sh
#   IMAGE=bio-is-curriculum:latest ./run_docker.sh 0 webkb 10
#   IMAGE=bio-is-curriculum:latest ./run_docker.sh 3 reuters 5
#
# Variáveis de ambiente opcionais:
#   MODEL      modelo a usar: lr | roberta         (default: roberta)
#   MODES      modos separados por espaço           (default: "baseline is cl is_cl")
#   FOLDS      folds separados por espaço           (default: todos do split file)
#   BETA       taxa de redundância BIOIS            (default: 0.3)
#   THETA      taxa de ruído BIOIS                  (default: 0.2)
#   EPOCHS     épocas treino único (baseline/is)    (default: 6)
#   EPOCHS_PP  épocas por fase (cl/is_cl)           (default: 2)
#   LR         learning rate                        (default: 2e-5)
#   CPUS       limite de CPUs do container          (default: 16)
#   MEMORY     limite de memória do container       (default: 32g)
set -euo pipefail

# ── Configurações ──────────────────────────────────────────────────────────────
GPU_ID="${1:-7}"
DATASET="${2:-webkb}"
N_SPLITS="${3:-10}"

MODEL="${MODEL:-roberta}"
MODES="${MODES:-baseline is cl is_cl}"
FOLDS="${FOLDS:-}"           # vazio = descoberta automática pelo run_experiment.py
BETA="${BETA:-0.3}"
THETA="${THETA:-0.2}"
EPOCHS="${EPOCHS:-6}"
EPOCHS_PP="${EPOCHS_PP:-2}"
LR="${LR:-2e-5}"

CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32g}"

IMAGE="${IMAGE:-bio-is-curriculum:latest}"
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(pwd)}"
CONTAINER_WORKDIR="/app"

# ── Monta o comando run_experiment.py ─────────────────────────────────────────
CMD="python run_experiment.py ${DATASET}"
CMD="${CMD} --n-splits ${N_SPLITS}"
CMD="${CMD} --modes ${MODES}"
CMD="${CMD} --model ${MODEL}"
CMD="${CMD} --beta ${BETA}"
CMD="${CMD} --theta ${THETA}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --epochs-per-phase ${EPOCHS_PP}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --data_dir datasets"
CMD="${CMD} --results-dir results"

# Folds opcionais (se vazio, run_experiment descobre todos automaticamente)
if [ -n "${FOLDS}" ]; then
  CMD="${CMD} --folds ${FOLDS}"
fi

# ── Resumo ─────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Image    : ${IMAGE}"
echo "  GPU      : device=${GPU_ID}"
echo "  Dataset  : ${DATASET}  |  N-splits: ${N_SPLITS}"
echo "  Modes    : ${MODES}"
echo "  Folds    : ${FOLDS:-auto (todos do split file)}"
echo "  Model    : ${MODEL}"
echo "  Host dir : ${HOST_PROJECT_DIR}"
echo "  Comando  : ${CMD}"
echo "============================================================"

# ── Execução ───────────────────────────────────────────────────────────────────
docker run --rm \
  --gpus "device=${GPU_ID}" \
  --cpus="${CPUS}" \
  --memory="${MEMORY}" \
  -v "${HOST_PROJECT_DIR}/datasets:${CONTAINER_WORKDIR}/datasets" \
  -v "${HOST_PROJECT_DIR}/results:${CONTAINER_WORKDIR}/results" \
  -w "${CONTAINER_WORKDIR}" \
  "${IMAGE}" \
  bash -c "${CMD}"

echo ""
echo "============================================================"
echo "  Experimento concluido."
echo "============================================================"
