"""CLI principal do biO-IS-Curriculum.

Suporta 4 modos (--mode) que formam a matriz IS x CL:

  baseline -- sem IS, sem CL (fine-tuning padrao no train completo)
  is       -- com IS, sem CL (reducao por BIOIS, treino unico no subset)
  cl       -- sem IS, com CL (BIOIS so gera sinais, curriculum no train completo)
  is_cl    -- com IS e CL  (reducao + curriculum no subset)

Todos os resultados sao gravados em results/<run_id>/ como CSVs/JSON
de facil leitura e comparacao entre modos.

Timings uniformes gravados em timings.csv para todos os modos:
  data_load_time_s    -- leitura de disco + encoding de labels
  preprocess_time_s   -- fit do BIOIS (0 para baseline)
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
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curriculum.biois_curriculum import BIOISCurriculum
from curriculum.models import LogisticRegressionModel
from data.loader import DatasetLoader
from iSel.biois import BIOIS
from results.run import RunRecorder
from data.rare_class_upsampling import upsample_min_per_class


BIOIS_STRATKFOLD_SPLITS = 5  # alinhado ao StratifiedKFold em BIOIS.fitting_alpha


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
):
    """Avalia o modelo apos treino de fase unica e grava phase_metrics."""
    t0 = time.perf_counter()
    preds = model.predict(X_eval)
    pred_time = time.perf_counter() - t0

    proba = model.predict_proba(X_eval)
    ent = np.array([stats.entropy(p) for p in proba])
    threshold = np.quantile(ent, 0.8)
    mask_hard = ent >= threshold
    hard_f1 = (
        float(f1_score(y_test[mask_hard], preds[mask_hard], average="macro"))
        if mask_hard.sum() > 0
        else float("nan")
    )

    row = {
        "phase": phase,
        "n_samples": int(len(y_test)),
        "n_iter": model.n_iter,
        "train_time_s": float(train_time),
        "pred_time_s": float(pred_time),
        "micro_f1": float(f1_score(y_test, preds, average="micro")),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "accuracy": float(accuracy_score(y_test, preds)),
        "hard_slice_macro_f1": hard_f1,
    }
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
        choices=["baseline", "is", "cl", "is_cl"],
        default="is_cl",
        help="Modo de execucao: baseline | is | cl | is_cl (default: is_cl)",
    )

    # BIOIS
    parser.add_argument("--beta", type=float, default=0.3, help="Taxa de reducao de redundancia do BIOIS")
    parser.add_argument("--theta", type=float, default=0.2, help="Taxa de reducao de ruido do BIOIS")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42)

    # Curriculum
    parser.add_argument("--curriculum-beta", dest="curriculum_beta", type=float, default=0.5)
    parser.add_argument(
        "--curriculum-q",
        dest="curriculum_q",
        type=float,
        nargs=3,
        default=(0.3, 0.6, 0.95),
        metavar=("Q_LOW", "Q_MID", "Q_HIGH"),
    )

    # Modelo
    parser.add_argument("--model", choices=["lr", "roberta"], default="roberta")
    parser.add_argument("--hf-model", dest="hf_model", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=6, help="Epocas para treino unico (baseline/is)")
    parser.add_argument("--epochs-per-phase", dest="epochs_per_phase", type=int, default=2,
                        help="Epocas por fase (cl/is_cl)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16)
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

    # Seeds globais antes de qualquer chamada estocastica (BIOIS, sklearn, etc.)
    random.seed(args.random_state)
    np.random.seed(args.random_state)

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
    # Os mesmos indices sao usados para fatiar X_train, y_train (para BIOIS)
    # E texts_train, y_texts_train (para o modelo de linguagem).
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=2018)
    for _train_idx, _val_idx in sss.split(X_train_raw, y_train_raw):
        continue  # usa a ultima divisao gerada

    X_train = X_train_raw[_train_idx]
    y_train = y_train_raw[_train_idx]
    X_val   = X_train_raw[_val_idx]
    y_val   = y_train_raw[_val_idx]

    texts_train = texts_val = None
    y_texts_train = None
    if texts_train_raw is not None:
        texts_train = [texts_train_raw[i] for i in _train_idx]
        texts_val = [texts_train_raw[i] for i in _val_idx]
        y_texts_train = y_texts_raw[_train_idx]

    needs_is_early = args.mode in ("is", "is_cl")
    needs_signals_early = args.mode in ("cl", "is_cl")
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
    needs_is = args.mode in ("is", "is_cl")
    needs_signals = args.mode in ("cl", "is_cl")
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

    recorder.log_timing("preprocess_time_s", preprocess_time)

    # ------------------------------------------------------------------ dispatcher
    model = _build_model(args, recorder)

    if args.mode == "baseline":
        y_tr = y_texts_train if y_texts_train is not None else y_train
        y_te = y_test_texts  if y_test_texts  is not None else y_test
        print(f"\n[baseline] Treinando {args.model} em {len(y_tr)} instancias por {args.epochs} epocas...")
        X_train_input = texts_train if texts_train else X_train
        X_test_input = texts_test if texts_test else X_test
        if hasattr(model, "set_phase"):
            model.set_phase("full")
        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = args.epochs
        t0_train = time.perf_counter()
        model.fit_stage(X_train_input, y_tr)
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        print(f"  Treino concluido em {train_time:.1f}s")
        preds, proba, metrics = _eval_single_stage(
            model, X_test_input, y_te, recorder, train_time=train_time,
        )

    elif args.mode == "is":
        idx = selector.sample_indices_
        X_sub = texts_train if texts_train else X_train
        X_train_input = [X_sub[i] for i in idx] if isinstance(X_sub, list) else X_sub[idx]
        X_test_input = texts_test if texts_test else X_test
        # Para RoBERTa usa labels da fonte de texto; para LR usa labels do TF-IDF
        y_src = y_texts_train if y_texts_train is not None else y_train
        y_te  = y_test_texts  if y_test_texts  is not None else y_test
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
        model.fit_stage(X_train_input, y_sub)
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        print(f"  Treino concluido em {train_time:.1f}s")
        preds, proba, metrics = _eval_single_stage(
            model, X_test_input, y_te, recorder, train_time=train_time,
        )

    else:
        # cl ou is_cl
        q_low, q_mid, q_high = args.curriculum_q

        # Para RoBERTa usa labels da fonte de texto; para LR usa labels do TF-IDF
        y_src = y_texts_train if y_texts_train is not None else y_train
        y_te  = y_test_texts  if y_test_texts  is not None else y_test

        if args.mode == "cl":
            cl_selector = selector
            X_cl = X_train
            y_cl = y_src
            texts_cl = texts_train
        else:  # is_cl
            idx = selector.sample_indices_
            cl_selector = _slice_selector_signals(selector, idx)
            X_cl = X_train[idx]
            y_cl = y_src[idx]
            texts_cl = [texts_train[i] for i in idx] if texts_train else None

        if args.mode == "is_cl":
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

        print(
            f"\n[{args.mode}] BIOISCurriculum em {len(y_cl)} instancias "
            f"(q={q_low}/{q_mid}/{q_high}, {args.epochs_per_phase} ep/fase)..."
        )

        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = args.epochs_per_phase

        curriculum = BIOISCurriculum(
            model=model,
            beta=args.curriculum_beta,
            q_low=q_low,
            q_mid=q_mid,
            q_high=q_high,
            random_state=args.random_state,
        )
        curriculum.fit(
            cl_selector,
            X_cl,
            y_cl,
            X_test=X_test,
            y_test=y_te,
            X_text=texts_cl,
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
    print(f"Accuracy final: {metrics.get('accuracy', float('nan')):.4f}")
    print(f"Resultados em : {recorder.run_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
