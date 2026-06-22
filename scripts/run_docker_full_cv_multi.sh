#!/usr/bin/env bash
# Executa full CV para multiplos datasets, multiplos modos e (opcionalmente)
# multiplos metodos/schemes de curriculum em uma unica chamada.
#
# Estrutura de resultados:
#   - modos non-CL (raw, is, bN): results/<exp_base>/<mode>_fold<k>/
#   - modos CL (cl, is_cl, is_continuos_cl): results/<exp_base>_<method>[_<scheme>]/
#         (uma pasta por combinacao curriculum-method x scheme para nao
#          sobrescrever, ja que cada pasta de fold e nomeada apenas pelo modo)
#
# Uso:
#   IMAGE=<imagem> ./scripts/run_docker_full_cv_multi.sh [GPU_ID] [DATASET_1 DATASET_2 ...]
#
# Exemplos:
#   # Defaults (todos os datasets, todos modos, todos os metodos de CL):
#   IMAGE=bio-is-curriculum:latest ./scripts/run_docker_full_cv_multi.sh 0
#
#   # Forcando datasets via args posicionais:
#   IMAGE=bio-is-curriculum:latest MODES="raw is cl is_cl b1" \
#     ./scripts/run_docker_full_cv_multi.sh 0 webkb reuters
#
#   # Subconjunto de metodos de curriculum e schemes SPCL:
#   CURRICULUM_METHODS="spcl_loss" SPCL_LOSS_SCHEMES="linear mixture" \
#     ./scripts/run_docker_full_cv_multi.sh 0
#
#   # Splits diferentes por dataset:
#   IMAGE=bio-is-curriculum:latest DATASET_SPLITS="webkb:10 reuters:5 mpqa:10" \
#     ./scripts/run_docker_full_cv_multi.sh 0
#
# Variaveis de ambiente opcionais:
#   DATASETS             lista de datasets separados por espaco
#                        (default: "webkb reuters90 mpqa 20ng yelp_reviews twitter sst1")
#   N_SPLITS             n-splits padrao para datasets sem override (default: 10)
#   DATASET_SPLITS       overrides no formato "dataset:n" (default: "reuters90:5")
#   MODEL                modelo: lr | roberta                  (default: roberta)
#   MODES                modos separados por espaco             (default: "raw is cl is_cl b1")
#   CURRICULUM_METHODS   metodos a varrer para modos CL/IS_CL
#                        (default: "biois_discrete spcl_soft spcl_loss")
#   SPCL_LOSS_SCHEMES    schemes a varrer para spcl_loss
#                        (default: "linear binary log mixture")
#   FOLDS                folds separados por espaco             (default: auto)
#   BETA                 taxa de redundancia BIOIS              (default: 0.3)
#   THETA                taxa de ruido BIOIS                    (default: 0.2)
#   EPOCHS               epocas treino unico (raw/is/bN)        (default: 6)
#   EPOCHS_PP            epocas por fase (cl/is_cl)             (default: 2)
#   BATCH_SIZE           batch de treino                        (default: 32)
#   LR                   learning rate                          (default: 2e-5)
#   CPUS                 limite de CPUs do container            (default: 16)
#   MEMORY               limite de memoria do container         (default: 32g)
set -euo pipefail

GPU_ID="${1:-7}"
shift || true

if [ "$#" -gt 0 ]; then
  DATASETS="$*"
else
  DATASETS="${DATASETS:-webkb reuters90 mpqa 20ng yelp_reviews twitter sst1}"
fi

N_SPLITS="${N_SPLITS:-10}"
DATASET_SPLITS="${DATASET_SPLITS:-reuters90:5}"

MODEL="${MODEL:-roberta}"
MODES="${MODES:-raw is cl is_cl b1}"
CURRICULUM_METHODS="${CURRICULUM_METHODS:-biois_discrete spcl_soft spcl_loss}"
SPCL_LOSS_SCHEMES="${SPCL_LOSS_SCHEMES:-linear binary log mixture}"
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

# Particiona MODES em (modos sem CL) e (modos com CL).
NON_CL_MODES=""
CL_MODES=""
for m in ${MODES}; do
  case "${m}" in
    cl|is_cl|is_continuos_cl)
      CL_MODES="${CL_MODES} ${m}"
      ;;
    *)
      NON_CL_MODES="${NON_CL_MODES} ${m}"
      ;;
  esac
done
NON_CL_MODES="$(echo ${NON_CL_MODES} | xargs)"
CL_MODES="$(echo ${CL_MODES} | xargs)"

run_in_container() {
  local cmd="$1"
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
job_idx=0

# Total = (datasets x non_cl_modes existentes) + (datasets x cl_modes x metodos x schemes)
total_datasets=$(wc -w <<<"${DATASETS}")
total_non_cl=$(wc -w <<<"${NON_CL_MODES:-x}")
[ -z "${NON_CL_MODES}" ] && total_non_cl=0
total_cl_modes=$(wc -w <<<"${CL_MODES:-x}")
[ -z "${CL_MODES}" ] && total_cl_modes=0
total_methods=$(wc -w <<<"${CURRICULUM_METHODS}")
total_schemes=$(wc -w <<<"${SPCL_LOSS_SCHEMES}")

# Numero de variantes por metodo: spcl_loss varre schemes; outros = 1.
cl_variants=0
for method in ${CURRICULUM_METHODS}; do
  if [ "${method}" = "spcl_loss" ]; then
    cl_variants=$((cl_variants + total_schemes))
  else
    cl_variants=$((cl_variants + 1))
  fi
done
total_jobs=$((total_datasets * total_non_cl + total_datasets * total_cl_modes * cl_variants))

echo "============================================================"
echo "  FULL CV multi-dataset / multi-mode / multi-method"
echo "============================================================"
echo "  Image              : ${IMAGE}"
echo "  GPU                : device=${GPU_ID}"
echo "  Datasets           : ${DATASETS}"
echo "  Non-CL modes       : ${NON_CL_MODES:-(nenhum)}"
echo "  CL modes           : ${CL_MODES:-(nenhum)}"
echo "  Curriculum methods : ${CURRICULUM_METHODS}"
echo "  SPCL loss schemes  : ${SPCL_LOSS_SCHEMES}"
echo "  Model              : ${MODEL}"
echo "  Folds              : ${FOLDS:-auto (todos do split file)}"
echo "  N-splits (def)     : ${N_SPLITS}"
echo "  Split overrides    : ${DATASET_SPLITS:-nenhum}"
echo "  Total jobs (est.)  : ${total_jobs}"
echo "  Host dir           : ${HOST_PROJECT_DIR}"
echo "============================================================"

# Helper: monta o experiment-id base por (dataset, n_splits).
make_exp_base() {
  local dataset="$1"
  local n="$2"
  # Timestamp grava-se no inicio para que rodadas paralelas/sequenciais
  # do mesmo script compartilhem o mesmo prefixo por dataset.
  echo "${dataset}-${n}cv-${RUN_TIMESTAMP}"
}

RUN_TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

build_cmd() {
  local dataset="$1"
  local n_splits="$2"
  local modes="$3"
  local experiment_id="$4"
  shift 4
  local extras="$*"

  local cmd="python run_experiment.py ${dataset}"
  cmd="${cmd} --n-splits ${n_splits}"
  cmd="${cmd} --modes ${modes}"
  cmd="${cmd} --model ${MODEL}"
  cmd="${cmd} --beta ${BETA}"
  cmd="${cmd} --theta ${THETA}"
  cmd="${cmd} --epochs ${EPOCHS}"
  cmd="${cmd} --epochs-per-phase ${EPOCHS_PP}"
  cmd="${cmd} --batch-size ${BATCH_SIZE}"
  cmd="${cmd} --lr ${LR}"
  cmd="${cmd} --data_dir datasets"
  cmd="${cmd} --results-dir results"
  cmd="${cmd} --experiment-id ${experiment_id}"
  if [ -n "${FOLDS}" ]; then
    cmd="${cmd} --folds ${FOLDS}"
  fi
  if [ -n "${extras}" ]; then
    cmd="${cmd} ${extras}"
  fi
  echo "${cmd}"
}

for dataset in ${DATASETS}; do
  dataset_n_splits="$(resolve_n_splits "${dataset}")"
  exp_base="$(make_exp_base "${dataset}" "${dataset_n_splits}")"

  # ------------------------------------------------------------------
  # 1) Modos non-CL (raw, is, bN) — uma rodada por dataset, sem varrer
  #    curriculum-method (irrelevante para esses modos).
  # ------------------------------------------------------------------
  if [ -n "${NON_CL_MODES}" ]; then
    job_idx=$((job_idx + 1))
    label="[${job_idx}/${total_jobs}] ${dataset} :: non-CL modes (${NON_CL_MODES})"
    cmd="$(build_cmd "${dataset}" "${dataset_n_splits}" "${NON_CL_MODES}" "${exp_base}")"
    echo ""
    echo "============================================================"
    echo "  ${label}"
    echo "  Command: ${cmd}"
    echo "============================================================"
    if ! run_in_container "${cmd}"; then
      failures+=("${dataset}:non_cl")
      echo "!! Falha em ${dataset} non-CL (continuando)."
    fi
  fi

  # ------------------------------------------------------------------
  # 2) Modos CL — varre (curriculum-method [x scheme]). Cada combinacao
  #    grava em um experiment-id distinto para nao sobrescrever folds.
  # ------------------------------------------------------------------
  if [ -n "${CL_MODES}" ]; then
    for method in ${CURRICULUM_METHODS}; do
      if [ "${method}" = "spcl_loss" ]; then
        SCHEMES="${SPCL_LOSS_SCHEMES}"
      else
        SCHEMES="_"  # placeholder unico
      fi
      for scheme in ${SCHEMES}; do
        job_idx=$((job_idx + 1))
        if [ "${scheme}" = "_" ]; then
          exp_id="${exp_base}_${method}"
          extras="--curriculum-method ${method}"
          label="[${job_idx}/${total_jobs}] ${dataset} :: ${method} :: modes=${CL_MODES}"
        else
          exp_id="${exp_base}_${method}_${scheme}"
          extras="--curriculum-method ${method} --curriculum-loss-scheme ${scheme}"
          label="[${job_idx}/${total_jobs}] ${dataset} :: ${method}/${scheme} :: modes=${CL_MODES}"
        fi
        cmd="$(build_cmd "${dataset}" "${dataset_n_splits}" "${CL_MODES}" "${exp_id}" ${extras})"
        echo ""
        echo "============================================================"
        echo "  ${label}"
        echo "  Command: ${cmd}"
        echo "============================================================"
        if ! run_in_container "${cmd}"; then
          failures+=("${dataset}:${method}${scheme:+:${scheme}}")
          echo "!! Falha em ${dataset} ${method} ${scheme} (continuando)."
        fi
      done
    done
  fi
done

echo ""
echo "============================================================"
if [ "${#failures[@]}" -eq 0 ]; then
  echo "  Execucao concluida com sucesso (${job_idx} jobs)."
else
  echo "  Execucao concluida com falhas em:"
  for f in "${failures[@]}"; do
    echo "    - ${f}"
  done
  exit 1
fi
echo "============================================================"
