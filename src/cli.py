"""CLI principal do biO-IS-Curriculum.

Suporta modos (--mode) que formam a matriz IS x CL:

  raw              -- sem IS, sem CL (fine-tuning padrao no train completo)
  is               -- com IS, sem CL (reducao por BIOIS, treino unico no subset)
  cl               -- sem IS, com CL (BIOIS so gera sinais, curriculum no train completo)
  is_cl            -- com IS e CL  (reducao + curriculum no subset)
  is_continuos_cl  -- alias IS+CL com metodo spcl_soft por default

Metodos de curriculum (--curriculum-method): biois_discrete, spcl_soft, spcl_loss.

Todos os resultados sao gravados em results/<run_id>/ como CSVs/JSON
de facil leitura e comparacao entre modos.

Timings uniformes gravados em timings.csv para todos os modos:
  data_load_time_s    -- leitura de disco + encoding de labels
  preprocess_time_s   -- fit do BIOIS (0 para raw)
  model_train_time_s  -- treino total do modelo (soma das fases)
  total_run_time_s    -- wall-clock fim a fim
"""
import argparse
import os
import random
import sys
import time
from collections import Counter
from types import SimpleNamespace

# Determinismo: tem que ser setado ANTES de importar torch/transformers.
# CUBLAS_WORKSPACE_CONFIG e exigido por torch p/ algumas GEMMs deterministicas.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curriculum.methods.registry import build_curriculum_kwargs, get_curriculum_method, resolve_method_id
from curriculum.models import LogisticRegressionModel
from data.loader import DatasetLoader
from iSel.biois import BIOIS
from results.metrics import build_phase_metrics_row
from results.run import RunRecorder
from data.rare_class_upsampling import upsample_min_per_class
from baselines import get_baseline, baseline_run_id


BIOIS_STRATKFOLD_SPLITS = 5  # alinhado ao StratifiedKFold em BIOIS.fitting_alpha

IS_MODES = frozenset({"is", "is_cl", "is_continuos_cl"})
CL_MODES = frozenset({"cl", "is_cl", "is_continuos_cl"})
IS_CL_MODES = frozenset({"is_cl", "is_continuos_cl"})


def _print_oversampling(stage: str, stats) -> None:
    if stats.n_added == 0:
        print(f"  Oversampling ({stage}): sem alteracoes (n={stats.n_before})")
    else:
        print(
            f"  Oversampling ({stage}): {stats.n_before} -> {stats.n_after} "
            f"(+{stats.n_added} instancias)"
        )



def _build_model(args, recorder: RunRecorder):
    """Instancia o CurriculumModel escolhido pelo usuario."""
    if args.model == "roberta":
        try:
            from curriculum.roberta_model import RobertaModel
        except ImportError as exc:
            print(
                "ERRO: torch/transformers nao estao instalados. "
                "Instale com: uv add torch transformers tqdm"
            )
            raise SystemExit(1) from exc

        return RobertaModel(
            model_name=args.hf_model,
            epochs_per_stage=args.epochs_per_phase,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            class_balanced_loss=args.class_balanced_loss,
            random_state=args.random_state,
            history_callback=recorder.log_train_step,
        )
    else:
        return LogisticRegressionModel(random_state=args.random_state)


def _eval_single_stage(
    model, X_eval, y_test, recorder: RunRecorder, phase: str = "full",
    train_time: float = float("nan"),
    hard_slice_quantile: float = 0.8,
):
    """Avalia o modelo apos treino de fase unica e grava phase_metrics."""
    t0 = time.perf_counter()
    proba = model.predict_proba(X_eval)
    preds = np.argmax(proba, axis=1)
    pred_time = time.perf_counter() - t0
    training_stats = {}
    if hasattr(model, "get_training_stats"):
        training_stats = model.get_training_stats()
    row = build_phase_metrics_row(
        phase=phase,
        y_true=y_test,
        y_pred=preds,
        proba=proba,
        n_iter=model.n_iter,
        train_time_s=train_time,
        pred_time_s=pred_time,
        hard_slice_quantile=hard_slice_quantile,
        training_stats=training_stats,
    )
    recorder.log_phase(row)
    return preds, proba, row


def _slice_selector_signals(selector, idx):
    """Cria um seletor virtual fatiado aos indices sobreviventes (modo is_cl).

    O curriculum precisa de um objeto com os mesmos atributos de sinal do BIOIS
    mas indexados ao subset reduzido, para que os quantis de entropia/redundancia
    sejam calculados sobre o mesmo universo que sera treinado.
    """
    return SimpleNamespace(
        _probaEveryone=selector._probaEveryone[idx],
        _y_proba_of_pred=selector._y_proba_of_pred[idx],
        _pred=selector._pred[idx],
    )


def main():
    parser = argparse.ArgumentParser(
        description="biO-IS-Curriculum: selecao de instancias + curriculum learning para classificacao textual."
    )
    parser.add_argument("dataset", type=str, help="Nome do dataset (ex: webkb)")
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", dest="n_splits", type=int, default=10)

    # Modo de execucao
    parser.add_argument(
        "--mode",
        choices=["raw", "is", "cl", "is_cl", "is_continuos_cl"],
        default="is_cl",
        help="Modo de execucao: raw | is | cl | is_cl | is_continuos_cl (default: is_cl). "
             "Ignorado quando --baseline N e fornecido.",
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=None,
        help="Indice de baseline da literatura (ver BASELINES.md). "
             "Quando fornecido, sobrescreve --mode e roda apenas o baseline N.",
    )

    # BIOIS
    parser.add_argument("--beta", type=float, default=0.3, help="Taxa de reducao de redundancia do BIOIS")
    parser.add_argument("--theta", type=float, default=0.2, help="Taxa de reducao de ruido do BIOIS")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)

    # Curriculum
    parser.add_argument(
        "--curriculum-method",
        dest="curriculum_method",
        type=str,
        default=None,
        help="Metodo de curriculum: biois_discrete | spcl_soft | spcl_loss "
             "(default: biois_discrete; is_continuos_cl usa spcl_soft).",
    )
    parser.add_argument("--curriculum-beta", dest="curriculum_beta", type=float, default=0.5)
    parser.add_argument(
        "--curriculum-q",
        dest="curriculum_q",
        type=float,
        nargs=3,
        default=(0.3, 0.6, 0.95),
        metavar=("Q_LOW", "Q_MID", "Q_HIGH"),
    )
    parser.add_argument(
        "--curriculum-n-steps",
        dest="curriculum_n_steps",
        type=int,
        default=6,
        help="Passos nominais do curriculum continuo/SPCL (default: 6).",
    )
    parser.add_argument(
        "--curriculum-alpha-decay",
        dest="curriculum_alpha_decay",
        type=float,
        default=10.0,
        help="Suavidade do soft-pacing SPCL (default: 10.0).",
    )
    parser.add_argument(
        "--curriculum-soft-lambda-init",
        dest="curriculum_soft_lambda_init",
        type=float,
        default=0.25,
        help="Lambda inicial do SPCL soft loss-driven (default: 0.25).",
    )
    parser.add_argument(
        "--curriculum-soft-lambda-growth",
        dest="curriculum_soft_lambda_growth",
        type=float,
        default=1.4,
        help="Fator de crescimento de lambda no SPCL soft (default: 1.4).",
    )
    parser.add_argument(
        "--curriculum-soft-lambda-max",
        dest="curriculum_soft_lambda_max",
        type=float,
        default=1.0,
        help="Lambda maximo no SPCL soft (default: 1.0).",
    )
    parser.add_argument(
        "--curriculum-soft-min-weight",
        dest="curriculum_soft_min_weight",
        type=float,
        default=1e-3,
        help="Peso minimo para manter amostra ativa no SPCL soft (default: 1e-3).",
    )
    parser.add_argument(
        "--curriculum-soft-stability-tol",
        dest="curriculum_soft_stability_tol",
        type=float,
        default=5e-3,
        help="Tolerancia para pular fases quase identicas no SPCL soft (default: 5e-3).",
    )
    parser.add_argument(
        "--curriculum-soft-saturation-patience",
        dest="curriculum_soft_saturation_patience",
        type=int,
        default=2,
        help="Paciencia de saturacao para early-stop de fases no SPCL soft (default: 2).",
    )
    parser.add_argument(
        "--curriculum-soft-max-effective-steps",
        dest="curriculum_soft_max_effective_steps",
        type=int,
        default=6,
        help="Limite de passos efetivos treinados no SPCL soft (default: 6).",
    )
    parser.add_argument(
        "--curriculum-loss-scheme",
        dest="curriculum_loss_scheme",
        type=str,
        choices=("binary", "linear", "log", "mixture"),
        default="linear",
        help="Self-paced function f(v;lambda) do SPCL canonico (default: linear).",
    )
    parser.add_argument(
        "--curriculum-lambda-init",
        dest="curriculum_lambda_init",
        type=float,
        default=0.5,
        help="Lambda inicial do SPCL canonico (default: 0.5; CE multiclass tipica).",
    )
    parser.add_argument(
        "--curriculum-lambda-step",
        dest="curriculum_lambda_step",
        type=float,
        default=0.5,
        help="Incremento aditivo mu de lambda por passo (Alg.1 SPCL, default: 0.5).",
    )
    parser.add_argument(
        "--curriculum-lambda-mult",
        dest="curriculum_lambda_mult",
        type=float,
        default=1.0,
        help="Multiplicador geometrico de lambda (default: 1.0 = usa lambda-step). "
             "Se >1.0, sobrescreve o passo aditivo (compat. com pacing legado).",
    )
    parser.add_argument(
        "--curriculum-lambda-max",
        dest="curriculum_lambda_max",
        type=float,
        default=None,
        help="Teto opcional de lambda no SPCL canonico (default: sem teto).",
    )
    parser.add_argument(
        "--curriculum-lambda2",
        dest="curriculum_lambda2",
        type=float,
        default=None,
        help="Segundo limiar do scheme mixture (lambda1 > lambda2 > 0; "
             "default: lambda_init / 2).",
    )
    parser.add_argument(
        "--curriculum-loss-prior-reliability",
        dest="curriculum_loss_prior_reliability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa reliability BIOIS no prior a do SPCL (default: True).",
    )
    parser.add_argument(
        "--curriculum-min-weight",
        dest="curriculum_min_weight",
        type=float,
        default=1e-3,
        help="Peso minimo para manter amostra no SPCL loss (default: 1e-3).",
    )
    parser.add_argument(
        "--curriculum-loss-recompute-every",
        dest="curriculum_loss_recompute_every",
        type=int,
        default=2,
        help="Recomputa losses do conjunto de treino completo a cada K steps "
             "do SPCL (default: 2). K=1 reproduz o comportamento antigo.",
    )

    # Modelo
    parser.add_argument("--model", choices=["lr", "roberta"], default="roberta")
    parser.add_argument("--hf-model", dest="hf_model", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=6, help="Epocas para treino unico (raw/is)")
    parser.add_argument("--epochs-per-phase", dest="epochs_per_phase", type=int, default=1,
                        help="Epocas por fase (cl/is_cl)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int, default=64)
    parser.add_argument("--max-length", dest="max_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-3,
        help="L2 sobre pesos não-LayerNorm (default 1e-3, estável neste projeto p/ RoBERTa).",
    )
    parser.add_argument(
        "--warmup-ratio",
        dest="warmup_ratio",
        type=float,
        default=0.06,
        help="Fração de steps com warmup linear (default 0.06 para fine-tune curto).",
    )
    parser.add_argument(
        "--class-balanced-loss",
        dest="class_balanced_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ativa peso por frequencia de classe na cross-entropy (default: True).",
    )
    parser.add_argument(
        "--hard-slice-quantile",
        dest="hard_slice_quantile",
        type=float,
        default=0.8,
        help="Quantil de entropia para hard-slice (default: 0.8 = top 20%%).",
    )

    # Resultados
    parser.add_argument("--results-dir", dest="results_dir", type=str, default="results")
    parser.add_argument(
        "--experiment-id", dest="experiment_id", type=str, default=None,
        help="ID do experimento grupo (ex: webkb-10cv-20260504-ab12cd). "
             "Se fornecido, salva em results/<experiment_id>/<mode>_fold<k>/",
    )
    parser.add_argument("--run-id", dest="run_id", type=str, default=None,
                        help="ID da execucao individual (ignorado se --experiment-id for fornecido)")

    args = parser.parse_args()

    if args.curriculum_method is None:
        args.curriculum_method = (
            "spcl_soft" if args.mode == "is_continuos_cl" else "biois_discrete"
        )
    args.curriculum_method = resolve_method_id(args.curriculum_method)

    # Seeds globais antes de qualquer chamada estocastica (BIOIS, sklearn, etc.)
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # --baseline N sobrescreve --mode. Valida e ja resolve a classe do registry
    # para falhar cedo se o indice nao existir.
    baseline_cls = None
    if args.baseline is not None:
        baseline_cls = get_baseline(args.baseline)
        args.mode = baseline_run_id(args.baseline)  # ex.: "b1"

    # ------------------------------------------------------------------ setup
    if args.experiment_id is not None:
        # Estrutura agrupada: results/<experiment_id>/<mode>_fold<k>/
        base_dir = os.path.join(args.results_dir, args.experiment_id)
        run_id = f"{args.mode}_fold{args.fold}"
        recorder = RunRecorder(base_dir=base_dir, run_id=run_id)
    else:
        # Estrutura legada: results/<mode>-<timestamp>-<hex6>/
        recorder = RunRecorder(base_dir=args.results_dir, run_id=args.run_id)
        if args.run_id is None:
            import shutil
            old_dir = recorder.run_dir
            new_id = f"{args.mode}-{recorder.run_id}"
            new_dir = os.path.join(args.results_dir, new_id)
            os.rename(old_dir, new_dir)
            recorder.run_dir = new_dir
            recorder.run_id = new_id

    print("=" * 50)
    print(f"run_id : {recorder.run_id}")
    print(f"mode   : {args.mode}")
    print(f"model  : {args.model} ({args.hf_model if args.model == 'roberta' else 'sklearn LR'})")
    print(f"results: {recorder.run_dir}")
    print("=" * 50)

    recorder.save_config(vars(args))
    t0_total = time.perf_counter()

    # ------------------------------------------------------------------ load data
    loader = DatasetLoader(data_dir=args.data_dir, dataset_name=args.dataset)

    t0_load = time.perf_counter()

    texts_train_raw = texts_test = None
    y_texts_raw = y_test_texts = None
    if args.model == "roberta":
        # Loader alinhado: TF-IDF e calculado em memoria a partir de texts.txt,
        # garantindo que linha i de X_train corresponda a texts_train[i] e
        # y_train[i]. Os arquivos svmlight pre-construidos usam uma ordem
        # diferente do split.pkl, o que corrompia o pareamento (texto, label)
        # quando BIOIS rodava sobre TF-IDF e o RoBERTa treinava nos textos.
        print("Carregando TF-IDF (recomputado dos textos para alinhamento)...")
        (
            X_train_raw,
            y_train_raw,
            X_test,
            y_test,
            texts_train_raw,
            texts_test,
        ) = loader.load_aligned_fold(args.fold, n_splits=args.n_splits)
        y_texts_raw = y_train_raw
        y_test_texts = y_test
        print(f"  X_train_raw: {X_train_raw.shape}  X_test: {X_test.shape}")
        print(f"  {len(texts_train_raw)} textos de treino raw, {len(texts_test)} de teste")
    else:
        print("Carregando TF-IDF (svmlight pre-construido)...")
        X_train_raw, y_train_raw, X_test, y_test = loader.load_tfidf_fold(args.fold)
        print(f"  X_train_raw: {X_train_raw.shape}  X_test: {X_test.shape}")

    # ---- StratifiedShuffleSplit aplicado UMA vez sobre os indices TF-IDF -----
    # Garante pelo menos 2 instâncias por classe para permitir o split estratificado.
    # Isso evita o erro "The least populated classes in y have only 1 member" no reuters90.
    X_train_raw, y_train_raw, st_raw, texts_train_raw = upsample_min_per_class(
        X_train_raw,
        y_train_raw,
        min_count=2,
        random_state=args.random_state,
        texts=texts_train_raw,
    )
    if y_texts_raw is not None:
        y_texts_raw = y_train_raw
    _print_oversampling("pre-split (estabilizacao de classes raras)", st_raw)

    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=2018)
    for _train_idx, _val_idx in sss.split(X_train_raw, y_train_raw):
        continue  # usa a ultima divisao gerada

    X_train = X_train_raw[_train_idx]
    y_train = y_train_raw[_train_idx]
    X_val   = X_train_raw[_val_idx]
    y_val   = y_train_raw[_val_idx]

    texts_train = texts_val = None
    y_texts_train = y_texts_val = None
    if texts_train_raw is not None:
        texts_train = [texts_train_raw[i] for i in _train_idx]
        texts_val = [texts_train_raw[i] for i in _val_idx]
        y_texts_train = y_texts_raw[_train_idx]
        y_texts_val = y_texts_raw[_val_idx]

    needs_is_early = args.mode in IS_MODES
    needs_signals_early = args.mode in CL_MODES or baseline_cls is not None
    run_biois_early = needs_is_early or needs_signals_early

    if run_biois_early:
        if texts_train is not None:
            X_train, y_train, st_biois, txs = upsample_min_per_class(
                X_train,
                y_train,
                min_count=BIOIS_STRATKFOLD_SPLITS,
                random_state=args.random_state,
                texts=texts_train,
            )
            texts_train = txs
            y_texts_train = np.asarray(y_train)
            _print_oversampling("pre-IS (treino vista pelo BIOIS)", st_biois)
        else:
            X_train, y_train, st_biois, _ = upsample_min_per_class(
                X_train,
                y_train,
                min_count=BIOIS_STRATKFOLD_SPLITS,
                random_state=args.random_state,
            )
            _print_oversampling("pre-IS (treino vista pelo BIOIS)", st_biois)

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    print(f"  Classes treino (TF-IDF): {Counter(y_train.tolist())}")
    if texts_train is not None:
        print(f"  {len(texts_train)} textos treino, {len(texts_val)} validacao, {len(texts_test)} teste")

    data_load_time = time.perf_counter() - t0_load
    recorder.log_timing("data_load_time_s", data_load_time)
    print(f"  Carregamento: {data_load_time:.1f}s")
    print("=" * 50)

    # ------------------------------------------------------------------ BIOIS
    needs_is = args.mode in IS_MODES
    needs_signals = args.mode in CL_MODES or baseline_cls is not None
    run_biois = needs_is or needs_signals

    selector = None
    preprocess_time = 0.0
    if run_biois:
        biois_beta = args.beta if needs_is else 0.0
        biois_theta = args.theta if needs_is else 0.0
        print(f"BIOIS (beta={biois_beta}, theta={biois_theta})...")
        selector = BIOIS(beta=biois_beta, theta=biois_theta, random_state=args.random_state)
        t0_is = time.perf_counter()
        selector.fit(X_train, y_train)
        preprocess_time = time.perf_counter() - t0_is
        recorder.log_timing("is_fit_time_s", preprocess_time)
        print(f"  Reducao: {selector.reduction_:.4f}  ({len(selector.sample_indices_)} instancias)")
        print(f"  Tempo IS: {preprocess_time:.1f}s")
        y_train_arr = np.asarray(y_train)
        selected_mask = np.zeros(len(y_train_arr), dtype=bool)
        selected_mask[selector.sample_indices_] = True
        removed_y = y_train_arr[~selected_mask]
        total_by_class = Counter(y_train_arr.tolist())
        removed_by_class = Counter(removed_y.tolist())
        if removed_by_class:
            print("  Remocao por classe:")
            for cls in sorted(total_by_class):
                removed = removed_by_class.get(cls, 0)
                total = total_by_class[cls]
                pct = (100.0 * removed / total) if total else 0.0
                print(f"    classe {cls}: {removed}/{total} ({pct:.1f}%)")

        if needs_is:
            recorder.save_instance_selection(
                n_train_before=len(y_train_arr),
                n_train_after=len(selector.sample_indices_),
                reduction=selector.reduction_,
                beta=biois_beta,
                theta=biois_theta,
                removed_by_class=dict(removed_by_class),
                total_by_class=dict(total_by_class),
            )
        else:
            recorder.save_instance_selection(
                n_train_before=len(y_train_arr),
                n_train_after=len(y_train_arr),
                reduction=0.0,
                beta=biois_beta,
                theta=biois_theta,
                removed_by_class={},
                total_by_class=dict(total_by_class),
            )

    recorder.log_timing("preprocess_time_s", preprocess_time)

    # ------------------------------------------------------------------ dispatcher
    model = _build_model(args, recorder)

    # Fixa num_labels uma vez no modelo a partir do espaco completo de classes
    # (train + test), para que fases do curriculum que nao contenham TODAS as
    # classes (comum em datasets long-tail como reuters90) nao causem
    # subinicializacao do classificador.
    y_all_for_labels = np.concatenate([
        np.asarray(y_texts_train if y_texts_train is not None else y_train),
        np.asarray(y_test_texts if y_test_texts is not None else y_test),
    ])
    if hasattr(model, "num_labels"):
        model.num_labels = int(np.max(y_all_for_labels)) + 1

    if args.mode == "raw":
        y_tr = y_texts_train if y_texts_train is not None else y_train
        y_te = y_test_texts  if y_test_texts  is not None else y_test
        print(f"\n[raw] Treinando {args.model} em {len(y_tr)} instancias por {args.epochs} epocas...")
        X_train_input = texts_train if texts_train else X_train
        X_val_input = texts_val if texts_val else X_val
        X_test_input = texts_test if texts_test else X_test
        y_val_input = y_texts_val if y_texts_val is not None else y_val
        if hasattr(model, "set_phase"):
            model.set_phase("full")
        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = args.epochs
        t0_train = time.perf_counter()
        model.fit_stage(X_train_input, y_tr, X_val=X_val_input, y_val=y_val_input)
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        print(f"  Treino concluido em {train_time:.1f}s")
        preds, proba, metrics = _eval_single_stage(
            model,
            X_test_input,
            y_te,
            recorder,
            train_time=train_time,
            hard_slice_quantile=args.hard_slice_quantile,
        )
        recorder.log_timing("metric_eval_time_s", float(metrics.get("pred_time_s", float("nan"))))

    elif args.mode == "is":
        idx = selector.sample_indices_
        X_sub = texts_train if texts_train else X_train
        X_train_input = [X_sub[i] for i in idx] if isinstance(X_sub, list) else X_sub[idx]
        X_val_input = texts_val if texts_val else X_val
        X_test_input = texts_test if texts_test else X_test
        # Para RoBERTa usa labels da fonte de texto; para LR usa labels do TF-IDF
        y_src = y_texts_train if y_texts_train is not None else y_train
        y_te  = y_test_texts  if y_test_texts  is not None else y_test
        y_val_input = y_texts_val if y_texts_val is not None else y_val
        y_sub = np.asarray(y_src[idx])
        assert len(y_sub) == (
            len(X_train_input)
            if isinstance(X_train_input, list)
            else X_train_input.shape[0]
        )
        X_train_input, y_sub, st_post_is, _ = upsample_min_per_class(
            X_train_input,
            y_sub,
            min_count=BIOIS_STRATKFOLD_SPLITS,
            random_state=args.random_state,
        )
        _print_oversampling("pos-IS (conjunto efetivamente treinado)", st_post_is)
        print(f"\n[is] Treinando {args.model} em {len(y_sub)} instancias por {args.epochs} epocas...")
        if hasattr(model, "set_phase"):
            model.set_phase("full")
        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = args.epochs
        t0_train = time.perf_counter()
        model.fit_stage(X_train_input, y_sub, X_val=X_val_input, y_val=y_val_input)
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        print(f"  Treino concluido em {train_time:.1f}s")
        preds, proba, metrics = _eval_single_stage(
            model,
            X_test_input,
            y_te,
            recorder,
            train_time=train_time,
            hard_slice_quantile=args.hard_slice_quantile,
        )
        recorder.log_timing("metric_eval_time_s", float(metrics.get("pred_time_s", float("nan"))))

    else:
        # cl, is_cl ou baseline da literatura (b1, b2, ...)
        q_low, q_mid, q_high = args.curriculum_q

        # Para RoBERTa usa labels da fonte de texto; para LR usa labels do TF-IDF
        y_src = y_texts_train if y_texts_train is not None else y_train
        y_te  = y_test_texts  if y_test_texts  is not None else y_test

        if args.mode in IS_CL_MODES:
            idx = selector.sample_indices_
            cl_selector = _slice_selector_signals(selector, idx)
            X_cl = X_train[idx]
            y_cl = y_src[idx]
            texts_cl = [texts_train[i] for i in idx] if texts_train else None
        else:
            # cl ou qualquer baseline da literatura: roda sobre o conjunto inteiro
            cl_selector = selector
            X_cl = X_train
            y_cl = y_src
            texts_cl = texts_train

        if args.mode in IS_CL_MODES:
            y_ic = np.asarray(y_cl)
            assert len(y_ic) == (
                len(texts_cl)
                if texts_cl is not None
                else X_cl.shape[0]
            )
            if texts_cl is not None:
                X_cl, y_ic, st_post_ic, txs_ic = upsample_min_per_class(
                    X_cl,
                    y_ic,
                    min_count=BIOIS_STRATKFOLD_SPLITS,
                    random_state=args.random_state,
                    texts=texts_cl,
                )
                texts_cl = txs_ic
            else:
                X_cl, y_ic, st_post_ic, _ = upsample_min_per_class(
                    X_cl,
                    y_ic,
                    min_count=BIOIS_STRATKFOLD_SPLITS,
                    random_state=args.random_state,
                )
            _print_oversampling("pos-IS (subset curriculo is_cl)", st_post_ic)
            y_cl = y_ic

            # O upsample duplicou linhas em X_cl/y_cl/texts_cl, mas cl_selector
            # ainda tem os signals do tamanho original. Estende-os pelos mesmos
            # indices duplicados para o curriculum casar shapes em _extract_signals.
            if st_post_ic.n_added > 0:
                dup = st_post_ic.dup_row_idx
                cl_selector = SimpleNamespace(
                    _probaEveryone=np.concatenate(
                        [cl_selector._probaEveryone, cl_selector._probaEveryone[dup]], axis=0
                    ),
                    _y_proba_of_pred=np.concatenate(
                        [cl_selector._y_proba_of_pred, cl_selector._y_proba_of_pred[dup]], axis=0
                    ),
                    _pred=np.concatenate(
                        [cl_selector._pred, cl_selector._pred[dup]], axis=0
                    ),
                )

        if baseline_cls is not None:
            CurriculumCls = baseline_cls
            cl_label = f"baseline {baseline_cls.INDEX} ({baseline_cls.NAME})"
            curriculum_kwargs = {
                "model": model,
                "beta": args.curriculum_beta,
                "q_low": q_low,
                "q_mid": q_mid,
                "q_high": q_high,
                "random_state": args.random_state,
                "hard_slice_quantile": args.hard_slice_quantile,
            }
        else:
            CurriculumCls = get_curriculum_method(args.curriculum_method)
            cl_label = f"{args.curriculum_method} ({CurriculumCls.__name__})"
            curriculum_kwargs = build_curriculum_kwargs(args.curriculum_method, args)
            curriculum_kwargs["model"] = model

        method_detail = (
            f"q={q_low}/{q_mid}/{q_high}"
            if args.curriculum_method == "biois_discrete"
            else f"n_steps={args.curriculum_n_steps}"
        )
        print(
            f"\n[{args.mode}] {cl_label} em {len(y_cl)} instancias "
            f"({method_detail}, {args.epochs_per_phase} ep/fase)..."
        )

        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = args.epochs_per_phase

        curriculum = CurriculumCls(**curriculum_kwargs)
        curriculum.fit(
            cl_selector,
            X_cl,
            y_cl,
            X_test=X_test,
            y_test=y_te,
            X_val=X_val,
            y_val=(y_texts_val if y_texts_val is not None else y_val),
            X_text=texts_cl,
            X_val_text=texts_val,
            X_test_text=texts_test,
            recorder=recorder,
        )
        model = curriculum.model_

        X_test_input = texts_test if texts_test else X_test
        proba = model.predict_proba(X_test_input)
        preds = np.argmax(proba, axis=1)
        metrics = curriculum.history_[-1] if curriculum.history_ else {}
        print("\nHistorico por fase:")
        for row in curriculum.history_:
            print(f"  {row}")

    # ------------------------------------------------------------------ finalize
    total_time = time.perf_counter() - t0_total
    recorder.log_timing("total_run_time_s", total_time)
    y_save = y_test_texts if y_test_texts is not None else y_test
    recorder.save_predictions(y_save, preds, proba)

    print("=" * 50)
    print(f"Concluido em {total_time:.1f}s")
    print(f"Macro-F1 final: {metrics.get('macro_f1', float('nan')):.4f}")
    print(f"Micro-F1 final: {metrics.get('micro_f1', float('nan')):.4f}")
    print(f"F1-weighted final: {metrics.get('f1_weighted', float('nan')):.4f}")
    print(f"Accuracy final: {metrics.get('accuracy', float('nan')):.4f}")
    print(f"Resultados em : {recorder.run_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
