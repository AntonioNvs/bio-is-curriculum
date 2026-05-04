"""Executor de experimentos multi-fold para biO-IS-Curriculum.

Roda todos os modos (baseline, is, cl, is_cl) em todos os folds
disponibles no split file, salvando os artefatos em:

    results/<experiment_id>/
        baseline_fold0/
        baseline_fold1/
        ...
        is_fold0/
        cl_fold0/
        is_cl_fold0/
        ...
        summary.csv   <- metricas com IC 95% por modo

Uso:
    python run_experiment.py webkb --n-splits 10 --model lr
    python run_experiment.py webkb --n-splits 5 --folds 0 1 2 --model roberta

Todos os argumentos extras (--beta, --theta, --epochs, --lr, etc.)
sao repassados diretamente ao main.py de cada execucao.
"""
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_folds(dataset: str, data_dir: str, n_splits: int) -> list[int]:
    """Retorna a lista de fold_ids disponiveis no split pickle."""
    pkl_path = os.path.join(data_dir, dataset, "splits", f"split_{n_splits}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Split file nao encontrado: {pkl_path}\n"
            f"Certifique-se de que o dataset '{dataset}' foi baixado e "
            f"que --n-splits={n_splits} esta correto."
        )
    df = pd.read_pickle(pkl_path)
    return sorted(df["fold_id"].tolist())


def _run_single(
    dataset: str,
    mode: str,
    fold: int,
    experiment_id: str,
    extra_args: list[str],
    results_dir: str,
) -> int:
    """Spawna 'python main.py ...' para um unico (mode, fold).

    Retorna o exit code do subprocesso.
    """
    cmd = [
        sys.executable, "main.py", dataset,
        "--mode", mode,
        "--fold", str(fold),
        "--experiment-id", experiment_id,
        "--results-dir", results_dir,
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"[{mode} | fold {fold}]  {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, check=False)
    return result.returncode


# ---------------------------------------------------------------------------
# CI aggregation
# ---------------------------------------------------------------------------

def _aggregate(experiment_dir: str, modes: list[str], folds: list[int]) -> pd.DataFrame:
    """Le as phase_metrics.csv de cada (mode, fold) e calcula IC 95%.

    Usa a ultima linha de phase_metrics.csv (performance final do modelo)
    como metrica representativa do fold.

    Retorna DataFrame com colunas:
        mode, metric, mean, std, ci_95_low, ci_95_high, n_folds
    """
    metrics_of_interest = ["micro_f1", "macro_f1", "accuracy", "hard_slice_macro_f1"]
    records: dict[str, dict[str, list[float]]] = {m: {k: [] for k in metrics_of_interest} for m in modes}

    missing: list[str] = []
    for mode in modes:
        for fold in folds:
            path = os.path.join(experiment_dir, f"{mode}_fold{fold}", "phase_metrics.csv")
            if not os.path.exists(path):
                missing.append(path)
                continue
            df = pd.read_csv(path)
            if df.empty:
                missing.append(path)
                continue
            last = df.iloc[-1]
            for k in metrics_of_interest:
                val = float(last.get(k, float("nan")))
                if not np.isnan(val):
                    records[mode][k].append(val)

    if missing:
        print(f"\nAVISO: {len(missing)} arquivo(s) nao encontrado(s) / vazios:")
        for p in missing:
            print(f"  {p}")

    rows = []
    for mode in modes:
        for metric in metrics_of_interest:
            vals = np.array(records[mode][metric])
            n = len(vals)
            if n == 0:
                rows.append({
                    "mode": mode, "metric": metric,
                    "mean": float("nan"), "std": float("nan"),
                    "ci_95_low": float("nan"), "ci_95_high": float("nan"),
                    "n_folds": 0,
                })
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
            if n > 1:
                t_crit = float(stats.t.ppf(0.975, df=n - 1))
                margin = t_crit * std / np.sqrt(n)
                ci_low = mean - margin
                ci_high = mean + margin
            else:
                ci_low = ci_high = float("nan")
            rows.append({
                "mode": mode, "metric": metric,
                "mean": mean, "std": std,
                "ci_95_low": ci_low, "ci_95_high": ci_high,
                "n_folds": n,
            })

    return pd.DataFrame(rows)


def _save_summary(summary_df: pd.DataFrame, experiment_dir: str) -> str:
    path = os.path.join(experiment_dir, "summary.csv")
    summary_df.to_csv(path, index=False, float_format="%.6f")
    return path


def _print_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("RESUMO DO EXPERIMENTO  (IC 95% via t-Student)")
    print("=" * 70)
    pivot = summary_df[summary_df["metric"] == "macro_f1"].set_index("mode")[
        ["mean", "ci_95_low", "ci_95_high", "n_folds"]
    ]
    print("Macro-F1:")
    print(pivot.to_string())
    print()
    pivot2 = summary_df[summary_df["metric"] == "micro_f1"].set_index("mode")[
        ["mean", "ci_95_low", "ci_95_high"]
    ]
    print("Micro-F1:")
    print(pivot2.to_string())
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Executa experimentos multi-fold para biO-IS-Curriculum.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Argumentos proprios do runner
    parser.add_argument("dataset", type=str, help="Nome do dataset (ex: webkb)")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "is", "cl", "is_cl"],
        default=["baseline", "is", "cl", "is_cl"],
        help="Modos a executar (default: todos os 4)",
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        default=None,
        help="Folds a usar. Se omitido, usa todos do split file.",
    )
    parser.add_argument("--n-splits", dest="n_splits", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--results-dir", dest="results_dir", type=str, default="results")
    parser.add_argument(
        "--experiment-id", dest="experiment_id", type=str, default=None,
        help="ID do experimento (default: {dataset}-{n_splits}cv-{timestamp}-{hex6})",
    )
    parser.add_argument(
        "--fail-fast", dest="fail_fast", action="store_true",
        help="Aborta ao primeiro erro de subprocesso",
    )

    # Todos os outros argumentos sao repassados ao main.py
    args, extra_args = parser.parse_known_args()

    # Descobre folds
    if args.folds is not None:
        folds = sorted(args.folds)
    else:
        folds = _discover_folds(args.dataset, args.data_dir, args.n_splits)
    print(f"Folds selecionados: {folds}  ({len(folds)} execucoes por modo)")
    print(f"Modos: {args.modes}")
    total_runs = len(args.modes) * len(folds)
    print(f"Total de execucoes: {total_runs}")

    # Cria experiment_id
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        short = uuid4().hex[:6]
        experiment_id = f"{args.dataset}-{args.n_splits}cv-{ts}-{short}"

    experiment_dir = os.path.join(args.results_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"\nexperiment_id : {experiment_id}")
    print(f"Diretorio     : {experiment_dir}")

    # Repassa --n-splits e --data_dir ao main.py (caso nao estejam em extra_args)
    forwarded = list(extra_args)
    if "--n-splits" not in forwarded and "-n-splits" not in forwarded:
        forwarded += ["--n-splits", str(args.n_splits)]
    if "--data_dir" not in forwarded:
        forwarded += ["--data_dir", args.data_dir]

    # ------------------------------------------------------------------ loop
    failed: list[tuple[str, int]] = []
    run_num = 0
    for mode in args.modes:
        for fold in folds:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}] mode={mode} fold={fold}")
            rc = _run_single(
                dataset=args.dataset,
                mode=mode,
                fold=fold,
                experiment_id=experiment_id,
                extra_args=forwarded,
                results_dir=args.results_dir,
            )
            if rc != 0:
                msg = f"mode={mode} fold={fold} saiu com codigo {rc}"
                print(f"ERRO: {msg}")
                failed.append((msg, rc))
                if args.fail_fast:
                    print("--fail-fast ativo, abortando.")
                    sys.exit(rc)

    # ------------------------------------------------------------------ aggregate
    print(f"\nAgregando resultados em {experiment_dir} ...")
    summary_df = _aggregate(experiment_dir, args.modes, folds)
    summary_path = _save_summary(summary_df, experiment_dir)
    _print_summary(summary_df)
    print(f"\nSummary salvo em: {summary_path}")

    if failed:
        print(f"\nAVISO: {len(failed)} execucao(oes) falharam:")
        for msg, _ in failed:
            print(f"  {msg}")
        sys.exit(1)

    print("\nExperimento concluido com sucesso.")


if __name__ == "__main__":
    main()
