#!/usr/bin/env bash
# Executa full CV para multiplos datasets e multiplos modos em uma unica chamada.
# Inspirado em run_docker_full_cv.sh, mas com loop por dataset.
#
# Uso:
#   IMAGE=<imagem> ./scripts/run_docker_full_cv_multi.sh [GPU_ID] [DATASET_1 DATASET_2 ...]
#
# Exemplos:
#   # Defaults (DATASETS env) com modos padrao:
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_full_cv_multi.sh 0
#
#   # Forcando datasets via args posicionais:
#   IMAGE=bio-is-curriculum:latest MODES="raw is cl is_cl b1" \
#     ./scripts/run_docker_full_cv_multi.sh 0 webkb reuters
#
#   # Splits diferentes por dataset:
#   IMAGE=bio-is-curriculum:latest DATASET_SPLITS="webkb:10 reuters:5 mpqa:10" \
#     ./scripts/run_docker_full_cv_multi.sh 0
#
# Variaveis de ambiente opcionais:
#   DATASETS        lista de datasets separados por espaco
#                   (default: "webkb reuters mpqa")
#   N_SPLITS        n-splits padrao para datasets sem override (default: 10)
#   DATASET_SPLITS  overrides no formato "dataset:n" separados por espaco
#                   (ex: "webkb:10 reuters:5")
#   MODEL           modelo: lr | roberta                  (default: roberta)
#   MODES           modos separados por espaco             (default: "raw is cl is_cl")
#   FOLDS           folds separados por espaco             (default: auto)
#   BETA            taxa de redundancia BIOIS              (default: 0.3)
#   THETA           taxa de ruido BIOIS                    (default: 0.2)
#   EPOCHS          epocas treino unico (raw/is)           (default: 6)
#   EPOCHS_PP       epocas por fase (cl/is_cl/bN)          (default: 2)
#   BATCH_SIZE      batch de treino                        (default: 32)
#   LR              learning rate                          (default: 2e-5)
#   CPUS            limite de CPUs do container            (default: 16)
#   MEMORY          limite de memoria do container         (default: 32g)
set -euo pipefail

GPU_ID="${1:-7}"
shift || true

if [ "$#" -gt 0 ]; then
  DATASETS="$*"
else
  DATASETS="${DATASETS:-webkb reuters90 mpqa ohsumed yelp_reviews twitter sst1}"
fi

N_SPLITS="${N_SPLITS:-10}"
DATASET_SPLITS="${DATASET_SPLITS:-}"

MODEL="${MODEL:-roberta}"
MODES="${MODES:-raw is cl is_cl b1}"
FOLDS="${FOLDS:-}" # vazio = descoberta automatica pelo run_experiment.py
BETA="${BETA:-0.3}"
THETA="${THETA:-0.2}"
EPOCHS="${EPOCHS:-6}"
EPOCHS_PP="${EPOCHS_PP:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-5}"

CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32g}"

IMAGE="${IMAGE:-bio-is-curriculum:latest}"
HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONTAINER_WORKDIR="/app"

resolve_n_splits() {
  local dataset="$1"
  local pair name n

  for pair in ${DATASET_SPLITS}; do
    name="${pair%%:*}"
    n="${pair#*:}"
    if [ "${name}" = "${dataset}" ]; then
      echo "${n}"
      return 0
    fi
  done

  echo "${N_SPLITS}"
}

failures=()
idx=0
total_datasets=$(wc -w <<<"${DATASETS}")

echo "============================================================"
echo "  FULL CV multi-dataset/multi-mode"
echo "============================================================"
echo "  Image          : ${IMAGE}"
echo "  GPU            : device=${GPU_ID}"
echo "  Datasets       : ${DATASETS}"
echo "  Modes          : ${MODES}"
echo "  Model          : ${MODEL}"
echo "  Folds          : ${FOLDS:-auto (todos do split file)}"
echo "  N-splits (def) : ${N_SPLITS}"
echo "  Overrides       : ${DATASET_SPLITS:-nenhum}"
echo "  Host dir       : ${HOST_PROJECT_DIR}"
echo "============================================================"

for dataset in ${DATASETS}; do
  idx=$((idx + 1))
  dataset_n_splits="$(resolve_n_splits "${dataset}")"

  CMD="python run_experiment.py ${dataset}"
  CMD="${CMD} --n-splits ${dataset_n_splits}"
  CMD="${CMD} --modes ${MODES}"
  CMD="${CMD} --model ${MODEL}"
  CMD="${CMD} --beta ${BETA}"
  CMD="${CMD} --theta ${THETA}"
  CMD="${CMD} --epochs ${EPOCHS}"
  CMD="${CMD} --epochs-per-phase ${EPOCHS_PP}"
  CMD="${CMD} --batch-size ${BATCH_SIZE}"
  CMD="${CMD} --lr ${LR}"
  CMD="${CMD} --data_dir datasets"
  CMD="${CMD} --results-dir results"

  if [ -n "${FOLDS}" ]; then
    CMD="${CMD} --folds ${FOLDS}"
  fi

  echo ""
  echo "============================================================"
  echo "  [${idx}/${total_datasets}] Dataset: ${dataset}  |  N-splits: ${dataset_n_splits}"
  echo "  Command: ${CMD}"
  echo "============================================================"

  if ! docker run --rm \
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
    bash -c "${CMD}"; then
    failures+=("${dataset}")
    echo "!! Falha em dataset '${dataset}' (continuando)."
  fi
done

echo ""
echo "============================================================"
if [ "${#failures[@]}" -eq 0 ]; then
  echo "  Execucao concluida com sucesso para todos os datasets."
else
  echo "  Execucao concluida com falhas em: ${failures[*]}"
  exit 1
fi
echo "============================================================"
