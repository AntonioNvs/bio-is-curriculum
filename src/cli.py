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
import sys
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curriculum.biois_curriculum import BIOISCurriculum
from curriculum.models import LogisticRegressionModel
from data.loader import DatasetLoader
from iSel.biois import BIOIS
from results.run import RunRecorder


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

    print("Carregando TF-IDF...")
    t0_load = time.perf_counter()
    X_train, y_train, X_test, y_test = loader.load_tfidf_fold(args.fold)
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  Classes treino: {Counter(y_train.tolist())}")

    texts_train = texts_test = None
    if args.model == "roberta":
        print("Carregando textos crus...")
        texts_train, _, texts_test, _ = loader.load_texts_fold(args.fold, n_splits=args.n_splits)
        print(f"  {len(texts_train)} textos de treino, {len(texts_test)} de teste")

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

    recorder.log_timing("preprocess_time_s", preprocess_time)

    # ------------------------------------------------------------------ dispatcher
    model = _build_model(args, recorder)

    if args.mode == "baseline":
        print(f"\n[baseline] Treinando {args.model} em {len(y_train)} instancias por {args.epochs} epocas...")
        X_train_input = texts_train if texts_train else X_train
        X_test_input = texts_test if texts_test else X_test
        if hasattr(model, "set_phase"):
            model.set_phase("full")
        if hasattr(model, "epochs_per_stage"):
            model.epochs_per_stage = args.epochs
        t0_train = time.perf_counter()
        model.fit_stage(X_train_input, y_train)
        train_time = time.perf_counter() - t0_train
        recorder.log_timing("model_train_time_s", train_time)
        print(f"  Treino concluido em {train_time:.1f}s")
        preds, proba, metrics = _eval_single_stage(
            model, X_test_input, y_test, recorder, train_time=train_time,
        )

    elif args.mode == "is":
        idx = selector.sample_indices_
        X_sub = texts_train if texts_train else X_train
        X_train_input = [X_sub[i] for i in idx] if isinstance(X_sub, list) else X_sub[idx]
        X_test_input = texts_test if texts_test else X_test
        y_sub = y_train[idx]
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
            model, X_test_input, y_test, recorder, train_time=train_time,
        )

    else:
        # cl ou is_cl
        q_low, q_mid, q_high = args.curriculum_q

        if args.mode == "cl":
            cl_selector = selector
            X_cl = X_train
            y_cl = y_train
            texts_cl = texts_train
        else:  # is_cl
            idx = selector.sample_indices_
            cl_selector = _slice_selector_signals(selector, idx)
            X_cl = X_train[idx]
            y_cl = y_train[idx]
            texts_cl = [texts_train[i] for i in idx] if texts_train else None

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
            y_test=y_test,
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
    recorder.save_predictions(y_test, preds, proba)

    print("=" * 50)
    print(f"Concluido em {total_time:.1f}s")
    print(f"Macro-F1 final: {metrics.get('macro_f1', float('nan')):.4f}")
    print(f"Micro-F1 final: {metrics.get('micro_f1', float('nan')):.4f}")
    print(f"Accuracy final: {metrics.get('accuracy', float('nan')):.4f}")
    print(f"Resultados em : {recorder.run_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
