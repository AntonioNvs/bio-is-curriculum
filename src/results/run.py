"""Gravacao estruturada de resultados de execucao.

Cada execucao gera um `run_id` unico e salva todos os artefatos em
`results/<run_id>/`, de modo que multiplas execucoes (4 modos, N seeds)
possam ser comparadas carregando os CSVs diretamente.
"""
import csv
import json
import os
import subprocess
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np
from scipy import stats


class RunRecorder:
    """Grava config, metricas, historico de treino, timings e predicoes
    em `<base_dir>/<run_id>/`.

    Arquivos gerados:
    - config.json            -- hyperparametros + dataset + fold + commit git
    - timings.csv            -- colunas: name, seconds
    - phase_metrics.csv      -- colunas: phase, n_samples, n_iter,
                                train_time_s, pred_time_s, micro_f1,
                                macro_f1, accuracy, hard_slice_macro_f1
    - train_history.csv      -- colunas: phase, epoch, step, loss, lr
    - predictions_test.csv   -- colunas: idx, y_true, y_pred, pred_entropy
    """

    PHASE_METRICS_COLS = [
        "phase", "n_samples", "n_iter",
        "train_time_s", "pred_time_s",
        "micro_f1", "macro_f1", "accuracy", "hard_slice_macro_f1",
    ]
    TRAIN_HISTORY_COLS = ["phase", "epoch", "step", "loss", "lr"]
    TIMINGS_COLS = ["name", "seconds"]
    PREDICTIONS_COLS = ["idx", "y_true", "y_pred", "pred_entropy"]

    def __init__(self, base_dir: str = "results", run_id: str | None = None):
        if run_id is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            short = uuid4().hex[:6]
            run_id = f"{ts}-{short}"
        self.run_id = run_id
        self.run_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._initialized_files: set[str] = set()

    def path(self, filename: str) -> str:
        """Retorna o caminho absoluto de um arquivo dentro de run_dir."""
        return os.path.join(self.run_dir, filename)

    def save_config(self, d: dict[str, Any]) -> None:
        """Salva config.json com todos os parametros da execucao."""
        config = dict(d)
        config["run_id"] = self.run_id
        config["timestamp"] = datetime.now().isoformat()
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            config["git_commit"] = result.stdout.strip()
        except Exception:
            config["git_commit"] = None

        with open(self.path("config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    def log_timing(self, name: str, seconds: float) -> None:
        """Appenda uma linha a timings.csv."""
        self._append_csv("timings.csv", self.TIMINGS_COLS, {"name": name, "seconds": seconds})

    def log_phase(self, row: dict[str, Any]) -> None:
        """Appenda uma linha a phase_metrics.csv."""
        self._append_csv("phase_metrics.csv", self.PHASE_METRICS_COLS, row)

    def log_train_step(self, row: dict[str, Any]) -> None:
        """Appenda uma linha a train_history.csv."""
        self._append_csv("train_history.csv", self.TRAIN_HISTORY_COLS, row)

    def save_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray | None = None,
        name: str = "predictions_test",
    ) -> None:
        """Escreve <name>.csv com y_true, y_pred e entropia preditiva."""
        filepath = self.path(f"{name}.csv")
        ent = (
            np.array([stats.entropy(p) for p in proba], dtype=np.float64)
            if proba is not None
            else np.full(len(y_true), float("nan"))
        )
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.PREDICTIONS_COLS)
            writer.writeheader()
            for i, (yt, yp, e) in enumerate(zip(y_true, y_pred, ent)):
                writer.writerow({"idx": i, "y_true": int(yt), "y_pred": int(yp), "pred_entropy": float(e)})

    # ------------------------------------------------------------------
    def _append_csv(self, filename: str, cols: list[str], row: dict[str, Any]) -> None:
        filepath = self.path(filename)
        write_header = filename not in self._initialized_files and not os.path.exists(filepath)
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._initialized_files.add(filename)
