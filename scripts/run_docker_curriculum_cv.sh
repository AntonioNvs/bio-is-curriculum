#!/usr/bin/env bash
# Full CV para os 3 metodos de curriculum x {cl, is_cl} em todos os datasets.
#
# Por dataset executa 6 configuracoes x n_folds:
#   biois_discrete | spcl_soft | spcl_loss  x  cl (sem IS) | is_cl (com IS)
#
# Cada metodo gera um experiment_id separado por dataset (evita sobrescrever
# resultados, pois pastas sao nomeadas apenas por modo: cl_foldK, is_cl_foldK).
#
# Uso:
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_curriculum_cv.sh [GPU_ID] [DATASET ...]
#
# Exemplos:
#   # Todos os datasets default, GPU 0:
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_curriculum_cv.sh 0
#
#   # Smoke test (1 fold, 1 dataset):
#   FOLDS="0" ./scripts/run_docker_curriculum_cv.sh 0 webkb
#
#   # Subconjunto de metodos:
#   CURRICULUM_METHODS="spcl_soft spcl_loss" ./scripts/run_docker_curriculum_cv.sh 0
#
# Variaveis de ambiente opcionais:
#   DATASETS             datasets separados por espaco (default: todos)
#   CURRICULUM_METHODS   metodos separados por espaco
#                        (default: "biois_discrete spcl_soft spcl_loss")
#   N_SPLITS             n-splits padrao (default: 10)
#   DATASET_SPLITS       overrides "dataset:n" (default inclui reuters90:5)
#   MODEL                lr | roberta (default: roberta)
#   FOLDS                folds explicitos (default: auto = todos do split file)
#   BETA / THETA         BIOIS (default: 0.3 / 0.2)
#   EPOCHS / EPOCHS_PP   epocas (default: 6 / 2)
#   BATCH_SIZE / LR      treino (default: 32 / 2e-5)
#   CPUS / MEMORY        limites do container (default: 16 / 32g)
#   IMAGE                imagem Docker (default: bio-is-curriculum:latest)
set -euo pipefail

GPU_ID="${1:-7}"
shift || true

if [ "$#" -gt 0 ]; then
  DATASETS="$*"
else
  DATASETS="${DATASETS:-webkb reuters90 mpqa ohsumed yelp_reviews twitter sst1}"
fi

CURRICULUM_METHODS="${CURRICULUM_METHODS:-biois_discrete spcl_soft spcl_loss}"
MODES="${MODES:-cl is_cl}"

N_SPLITS="${N_SPLITS:-10}"
DATASET_SPLITS="${DATASET_SPLITS:-reuters90:5}"

MODEL="${MODEL:-roberta}"
FOLDS="${FOLDS:-}"
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

run_dataset() {
  local method="$1"
  local dataset="$2"
  local dataset_n_splits="$3"
  local label="$4"

  local cmd="python run_experiment.py ${dataset}"
  cmd="${cmd} --n-splits ${dataset_n_splits}"
  cmd="${cmd} --modes ${MODES}"
  cmd="${cmd} --model ${MODEL}"
  cmd="${cmd} --curriculum-method ${method}"
  cmd="${cmd} --beta ${BETA}"
  cmd="${cmd} --theta ${THETA}"
  cmd="${cmd} --epochs ${EPOCHS}"
  cmd="${cmd} --epochs-per-phase ${EPOCHS_PP}"
  cmd="${cmd} --batch-size ${BATCH_SIZE}"
  cmd="${cmd} --lr ${LR}"
  cmd="${cmd} --data_dir datasets"
  cmd="${cmd} --results-dir results"

  if [ -n "${FOLDS}" ]; then
    cmd="${cmd} --folds ${FOLDS}"
  fi

  echo ""
  echo "============================================================"
  echo "  ${label}"
  echo "  Command: ${cmd}"
  echo "============================================================"

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
    bash -c "${cmd}"
}

failures=()
total_methods=$(wc -w <<<"${CURRICULUM_METHODS}")
total_datasets=$(wc -w <<<"${DATASETS}")
total_jobs=$((total_methods * total_datasets))
job_idx=0

echo "============================================================"
echo "  CURRICULUM METHODS FULL CV"
echo "============================================================"
echo "  Image              : ${IMAGE}"
echo "  GPU                : device=${GPU_ID}"
echo "  Curriculum methods : ${CURRICULUM_METHODS}"
echo "  Modes (IS x CL)    : ${MODES}"
echo "  Datasets           : ${DATASETS}"
echo "  Model              : ${MODEL}"
echo "  Folds              : ${FOLDS:-auto (todos do split file)}"
echo "  N-splits (def)     : ${N_SPLITS}"
echo "  Split overrides    : ${DATASET_SPLITS:-nenhum}"
echo "  Total jobs         : ${total_jobs} (${total_methods} metodos x ${total_datasets} datasets)"
echo "  Host dir           : ${HOST_PROJECT_DIR}"
echo "============================================================"

for method in ${CURRICULUM_METHODS}; do
  echo ""
  echo "############################################################"
  echo "  CURRICULUM METHOD: ${method}"
  echo "############################################################"

  ds_idx=0
  for dataset in ${DATASETS}; do
    ds_idx=$((ds_idx + 1))
    job_idx=$((job_idx + 1))
    dataset_n_splits="$(resolve_n_splits "${dataset}")"
    label="[${job_idx}/${total_jobs}] method=${method} dataset=${dataset} (${ds_idx}/${total_datasets}) n-splits=${dataset_n_splits}"

    if ! run_dataset "${method}" "${dataset}" "${dataset_n_splits}" "${label}"; then
      failures+=("${method}:${dataset}")
      echo "!! Falha em ${method} / ${dataset} (continuando)."
    fi
  done
done

echo ""
echo "============================================================"
if [ "${#failures[@]}" -eq 0 ]; then
  echo "  Execucao concluida com sucesso."
else
  echo "  Execucao concluida com falhas em:"
  for f in "${failures[@]}"; do
    echo "    - ${f}"
  done
  exit 1
fi
echo "============================================================"
