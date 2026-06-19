"""Entry point unico para experimentos biO-IS-Curriculum.

Um comando. Um YAML. Resultado completo com sumario automatico.

Uso:
    python run.py experiments/webkb.yaml
    python run.py experiments/webkb.yaml --folds 0 1 2   # override de folds
    python run.py experiments/webkb.yaml --verbose        # mostra stdout dos subprocessos

O YAML define dataset, modos, curriculum, hiperparametros. Campos omitidos
usam defaults sensatos (via DEFAULTS abaixo). Exemplo minimo:

    dataset: webkb
    n_splits: 10
    modes: [raw, is, cl, is_cl]
    instance_selection: {beta: 0.3, theta: 0.2}
    curriculum: {method: biois_discrete}

Para cada (mode, fold), spawna `python main.py ...` como subprocesso
(isolamento de GPU entre runs). Ao final, agrega metricas com IC 95%
e imprime o sumario no terminal.

Principio KISS: A UNICA interface que o usuario precisa conhecer.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# ---------------------------------------------------------------------------
# Defaults (mesmos valores do cli.py argparse, consolidados aqui).
# ---------------------------------------------------------------------------

DEFAULTS = {
    "model": "roberta",
    "hf_model": "roberta-base",
    "results_dir": "results",
    "data_dir": "datasets",
    "random_state": 42,
    "hard_slice_quantile": 0.8,
    # Instance selection
    "beta": 0.3,
    "theta": 0.2,
    # Training
    "epochs": 6,
    "epochs_per_phase": 2,
    "batch_size": 32,
    "eval_batch_size": 64,
    "max_length": 256,
    "lr": 2.0e-5,
    "weight_decay": 1.0e-3,
    "warmup_ratio": 0.06,
    "class_balanced_loss": True,
    # Curriculum
    "curriculum_method": "biois_discrete",
    "curriculum_beta": 0.5,
    "curriculum_q_low": 0.3,
    "curriculum_q_mid": 0.6,
    "curriculum_q_high": 0.95,
    "curriculum_n_steps": 6,
    "curriculum_alpha_decay": 10.0,
    "curriculum_soft_lambda_init": 0.25,
    "curriculum_soft_lambda_growth": 1.4,
    "curriculum_soft_lambda_max": 1.0,
    "curriculum_soft_min_weight": 1e-3,
    "curriculum_soft_stability_tol": 5e-3,
    "curriculum_soft_saturation_patience": 2,
    "curriculum_soft_max_effective_steps": 6,
    "curriculum_loss_scheme": "linear",
    "curriculum_lambda_init": 0.5,
    "curriculum_lambda_step": 0.5,
    "curriculum_lambda_mult": 1.0,
    "curriculum_lambda_max": None,
    "curriculum_lambda2": None,
    "curriculum_loss_prior_reliability": True,
    "curriculum_min_weight": 1e-3,
    "curriculum_loss_recompute_every": 2,
}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    """Carrega YAML. Tenta PyYAML; se nao disponivel, da mensagem clara."""
    try:
        import yaml
    except ImportError:
        sys.exit(
            "PyYAML nao encontrado. Instale com:  uv add pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge_defaults(config: dict) -> dict:
    """Preenche campos ausentes do config com DEFAULTS, deep-merge simples."""
    merged = dict(DEFAULTS)
    # Top-level
    for k in ("dataset", "n_splits", "model", "hf_model", "modes",
              "results_dir", "data_dir", "random_state", "hard_slice_quantile",
              "experiment_id"):
        if k in config:
            merged[k] = config[k]

    # Instance selection
    if "instance_selection" in config:
        is_cfg = config["instance_selection"]
        for k in ("beta", "theta"):
            if k in is_cfg:
                merged[k] = is_cfg[k]

    # Training
    if "training" in config:
        t_cfg = config["training"]
        for k in ("epochs", "epochs_per_phase", "batch_size", "eval_batch_size",
                  "max_length", "lr", "weight_decay", "warmup_ratio",
                  "class_balanced_loss"):
            if k in t_cfg:
                merged[k] = t_cfg[k]

    # Curriculum
    if "curriculum" in config:
        c_cfg = config["curriculum"]
        for k in ("method", "beta", "q_low", "q_mid", "q_high",
                  "n_steps", "alpha_decay",
                  "soft_lambda_init", "soft_lambda_growth", "soft_lambda_max",
                  "soft_min_weight", "soft_stability_tol",
                  "soft_saturation_patience", "soft_max_effective_steps",
                  "loss_scheme", "lambda_init", "lambda_step", "lambda_mult",
                  "lambda_max", "lambda2", "loss_prior_reliability",
                  "min_weight", "loss_recompute_every"):
            if k in c_cfg:
                merged[f"curriculum_{k}"] = c_cfg[k]
        # curriculum.method -> curriculum_method
        if "method" in c_cfg:
            merged["curriculum_method"] = c_cfg["method"]

    return merged


# ---------------------------------------------------------------------------
# CLI builder
# ---------------------------------------------------------------------------

def _build_cli_args(cfg: dict, mode: str, fold: int, experiment_id: str, results_dir: str) -> list[str]:
    """Converte config dict + mode + fold para lista de argumentos do main.py."""
    args = [
        sys.executable, "main.py", cfg["dataset"],
        "--fold", str(fold),
        "--experiment-id", experiment_id,
        "--results-dir", results_dir,
        "--n-splits", str(cfg["n_splits"]),
        "--data_dir", cfg["data_dir"],
        "--model", cfg["model"],
        "--random-state", str(cfg["random_state"]),
        "--hard-slice-quantile", str(cfg["hard_slice_quantile"]),
    ]

    # --mode ou --baseline
    baseline_match = re.match(r"^b([0-9]+)$", mode)
    if baseline_match:
        args += ["--baseline", baseline_match.group(1)]
    else:
        args += ["--mode", mode]

    # BIOIS
    if mode not in ("raw",):
        args += ["--beta", str(cfg["beta"]), "--theta", str(cfg["theta"])]

    # Training
    args += [
        "--epochs", str(cfg["epochs"]),
        "--epochs-per-phase", str(cfg["epochs_per_phase"]),
        "--batch-size", str(cfg["batch_size"]),
        "--eval-batch-size", str(cfg["eval_batch_size"]),
        "--max-length", str(cfg["max_length"]),
        "--lr", str(cfg["lr"]),
        "--weight-decay", str(cfg["weight_decay"]),
        "--warmup-ratio", str(cfg["warmup_ratio"]),
    ]
    if cfg["model"] == "roberta":
        args += ["--hf-model", cfg["hf_model"]]
    if not cfg["class_balanced_loss"]:
        args.append("--no-class-balanced-loss")

    # Curriculum
    if mode not in ("raw", "is"):
        args += ["--curriculum-method", cfg["curriculum_method"]]
        args += ["--curriculum-beta", str(cfg["curriculum_beta"])]

        method = cfg["curriculum_method"]
        if method == "biois_discrete":
            args += [
                "--curriculum-q",
                str(cfg["curriculum_q_low"]),
                str(cfg["curriculum_q_mid"]),
                str(cfg["curriculum_q_high"]),
            ]
        elif method == "spcl_soft":
            args += [
                "--curriculum-n-steps", str(cfg["curriculum_n_steps"]),
                "--curriculum-alpha-decay", str(cfg["curriculum_alpha_decay"]),
                "--curriculum-soft-lambda-init", str(cfg["curriculum_soft_lambda_init"]),
                "--curriculum-soft-lambda-growth", str(cfg["curriculum_soft_lambda_growth"]),
                "--curriculum-soft-lambda-max", str(cfg["curriculum_soft_lambda_max"]),
                "--curriculum-soft-min-weight", str(cfg["curriculum_soft_min_weight"]),
                "--curriculum-soft-stability-tol", str(cfg["curriculum_soft_stability_tol"]),
                "--curriculum-soft-saturation-patience", str(cfg["curriculum_soft_saturation_patience"]),
                "--curriculum-soft-max-effective-steps", str(cfg["curriculum_soft_max_effective_steps"]),
            ]
        elif method == "spcl_loss":
            args += [
                "--curriculum-n-steps", str(cfg["curriculum_n_steps"]),
                "--curriculum-loss-scheme", cfg["curriculum_loss_scheme"],
                "--curriculum-lambda-init", str(cfg["curriculum_lambda_init"]),
                "--curriculum-lambda-step", str(cfg["curriculum_lambda_step"]),
                "--curriculum-lambda-mult", str(cfg["curriculum_lambda_mult"]),
                "--curriculum-min-weight", str(cfg["curriculum_min_weight"]),
                "--curriculum-loss-recompute-every", str(cfg["curriculum_loss_recompute_every"]),
            ]
            if cfg["curriculum_lambda_max"] is not None:
                args += ["--curriculum-lambda-max", str(cfg["curriculum_lambda_max"])]
            if cfg["curriculum_lambda2"] is not None:
                args += ["--curriculum-lambda2", str(cfg["curriculum_lambda2"])]
            if not cfg["curriculum_loss_prior_reliability"]:
                args.append("--no-curriculum-loss-prior-reliability")

    return args


# ---------------------------------------------------------------------------
# Fold discovery
# ---------------------------------------------------------------------------

def _discover_folds(dataset: str, data_dir: str, n_splits: int) -> list[int]:
    import pandas as pd
    pkl_path = os.path.join(data_dir, dataset, "splits", f"split_{n_splits}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Split file nao encontrado: {pkl_path}\n"
            f"Certifique-se de que o dataset '{dataset}' foi baixado."
        )
    df = pd.read_pickle(pkl_path)
    return sorted(df["fold_id"].tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="biO-IS-Curriculum: entry point unico com YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config", type=str,
        help="Caminho do YAML de configuracao do experimento.",
    )
    parser.add_argument(
        "--folds", nargs="*", type=int, default=None,
        help="Folds a usar. Se omitido, usa todos do split file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Mostra stdout/stderr dos subprocessos em tempo real.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Aborta ao primeiro erro.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ carrega config
    config = _load_yaml(args.config)
    config.setdefault("modes", ["raw", "is", "cl", "is_cl"])
    cfg = _merge_defaults(config)

    dataset = cfg["dataset"]
    n_splits = cfg["n_splits"]
    modes = cfg["modes"]

    # ------------------------------------------------------------------ folds
    if args.folds is not None:
        folds = sorted(args.folds)
    else:
        folds = _discover_folds(dataset, cfg["data_dir"], n_splits)

    total_runs = len(modes) * len(folds)

    # ------------------------------------------------------------------ experiment id
    experiment_id = cfg.get("experiment_id")
    if not experiment_id:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        short = uuid4().hex[:6]
        experiment_id = f"{dataset}-{n_splits}cv-{ts}-{short}"

    results_dir = cfg["results_dir"]
    experiment_dir = os.path.join(results_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # ------------------------------------------------------------------ banner
    print("=" * 65)
    print("  biO-IS-Curriculum")
    print("=" * 65)
    print(f"  Config  : {args.config}")
    print(f"  Dataset : {dataset}  ({n_splits}-fold CV)")
    print(f"  Model   : {cfg['model']}")
    print(f"  Modes   : {', '.join(modes)}")
    print(f"  Folds   : {folds}  ({len(folds)} folds)")
    print(f"  Total   : {total_runs} execucoes")
    if "is_cl" in modes or "cl" in modes:
        print(f"  Curriculum: {cfg['curriculum_method']}")
    print(f"  Exp ID  : {experiment_id}")
    print(f"  Results : {experiment_dir}")
    print("=" * 65)

    # ------------------------------------------------------------------ executa
    failed: list[tuple[str, int, str]] = []
    run_num = 0
    for mode in modes:
        for fold in folds:
            run_num += 1
            label = f"[{run_num}/{total_runs}] {mode} fold={fold}"
            print(f"\n{label} ", end="", flush=True)

            cli_args = _build_cli_args(cfg, mode, fold, experiment_id, results_dir)

            if args.verbose:
                print()
                print(f"  {' '.join(cli_args)}")
                result = subprocess.run(cli_args, check=False)
            else:
                result = subprocess.run(
                    cli_args, check=False,
                    capture_output=True, text=True,
                )

            if result.returncode != 0:
                print(f"❌ FAIL (exit {result.returncode})")
                stderr_tail = ""
                if not args.verbose and result.stderr:
                    lines = result.stderr.strip().splitlines()
                    stderr_tail = "\n".join(lines[-5:])  # ultimas 5 linhas
                failed.append((f"{mode}_fold{fold}", result.returncode, stderr_tail))
                if args.fail_fast:
                    print("--fail-fast ativo, abortando.")
                    if stderr_tail:
                        print(stderr_tail)
                    sys.exit(result.returncode)
            else:
                print("✅")

    # ------------------------------------------------------------------ aggregate
    if not failed or not args.fail_fast:
        print(f"\n{'=' * 65}")
        print("Agregando resultados...")
        print(f"{'=' * 65}")

        # Reusa a funcao de agregacao do run_experiment.py
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_experiment import _aggregate, _print_summary

        summary_df = _aggregate(experiment_dir, modes, folds)
        summary_path = os.path.join(experiment_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")
        _print_summary(summary_df)
        print(f"\nSummary: {summary_path}")

    # ------------------------------------------------------------------ report failures
    if failed:
        print(f"\n⚠️  {len(failed)}/{total_runs} execucoes falharam:")
        for name, rc, stderr_tail in failed:
            print(f"  {name} (exit {rc})")
            if stderr_tail:
                for line in stderr_tail.splitlines():
                    print(f"    {line}")
        sys.exit(1)

    print("\n✅ Experimento concluido.")
    print(f"   Resultados: {experiment_dir}/")
    print(f"   Comando para re-summarize: python summary.py --metric macro_f1 {experiment_id}")


if __name__ == "__main__":
    main()
