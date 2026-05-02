"""Backend RoBERTa para o curriculum learning.

Implementa `CurriculumModel` usando `roberta-base` (ou qualquer checkpoint
do HuggingFace Hub compativel com `AutoModelForSequenceClassification`).

O loop de treino e feito manualmente (sem `Trainer`) para:
- suportar `sample_weight` por instancia;
- nao exigir `accelerate`;
- manter warm start real entre fases (o modelo nao e re-instanciado,
  apenas o optimizer/scheduler sao renovados a cada `fit_stage`).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.curriculum.models import CurriculumModel


class _TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray, weights: np.ndarray):
        self.encodings = encodings
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item


class RobertaModel(CurriculumModel):
    """Modelo RoBERTa fine-tunado de forma faseada (warm start entre fases).

    Parameters
    ----------
    model_name : str
        Nome do checkpoint HuggingFace (default: ``"roberta-base"``).

    num_labels : int, optional
        Numero de classes. Se ``None``, inferido de ``y`` no primeiro
        ``fit_stage``.

    epochs_per_stage : int
        Epocas de treino por chamada de ``fit_stage``.

    batch_size : int
        Tamanho de batch de treino.

    eval_batch_size : int
        Tamanho de batch para inferencia.

    max_length : int
        Comprimento maximo de tokenizacao (trunca textos maiores).

    lr : float
        Learning rate do AdamW.

    weight_decay : float
        Weight decay do AdamW.

    warmup_ratio : float
        Fracao de steps para warmup linear (por fase).

    device : str, optional
        ``"cuda"``, ``"cpu"`` ou ``None`` para autodeteccao.

    random_state : int
        Seed para reproducibilidade.

    history_callback : callable, optional
        Chamado a cada step com ``(phase_name, step, epoch, loss, lr)``.
        Use para gravar ``train_history.csv`` via ``RunRecorder``.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int | None = None,
        epochs_per_stage: int = 2,
        batch_size: int = 16,
        eval_batch_size: int = 64,
        max_length: int = 256,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        device: str | None = None,
        random_state: int = 42,
        history_callback: Callable | None = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.epochs_per_stage = epochs_per_stage
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.random_state = random_state
        self.history_callback = history_callback

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._tokenizer = None
        self._model = None
        self.global_step_: int = 0
        self._current_phase: str = "unknown"

    # ------------------------------------------------------------------
    # CurriculumModel interface
    # ------------------------------------------------------------------

    def fit_stage(self, texts: list[str], y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Continua (ou inicia) o fine-tuning por `epochs_per_stage` epocas."""
        torch.manual_seed(self.random_state + self.global_step_)

        n = len(y)
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64)
        sample_weight = np.array(sample_weight, dtype=np.float64)

        num_labels = int(np.max(y)) + 1
        self._lazy_init(num_labels)

        encodings = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = _TextDataset(encodings, y.astype(np.int64), sample_weight)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_steps = len(loader) * self.epochs_per_stage
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self._model.train()
        for epoch in range(self.epochs_per_stage):
            epoch_loss = 0.0
            for batch in tqdm(loader, desc=f"[{self._current_phase}] epoch {epoch + 1}/{self.epochs_per_stage}", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                weights = batch["weights"].to(self.device)

                optimizer.zero_grad()
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss_per_sample = F.cross_entropy(logits, labels, reduction="none")
                loss = (loss_per_sample * weights).mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                self.global_step_ += 1
                epoch_loss += loss.item()

                if self.history_callback is not None:
                    self.history_callback({
                        "phase": self._current_phase,
                        "epoch": epoch + 1,
                        "step": self.global_step_,
                        "loss": round(loss.item(), 6),
                        "lr": scheduler.get_last_lr()[0],
                    })

        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self._lazy_init(self.num_labels)
        self._model.eval()

        encodings = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = _TextDataset(
            encodings,
            np.zeros(len(texts), dtype=np.int64),
            np.ones(len(texts), dtype=np.float64),
        )
        loader = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False)

        all_proba = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                proba = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_proba.append(proba)

        return np.concatenate(all_proba, axis=0)

    @property
    def n_iter(self) -> int:
        return self.global_step_

    # ------------------------------------------------------------------

    def set_phase(self, phase_name: str) -> None:
        """Informa ao modelo qual fase esta sendo treinada (para logs)."""
        self._current_phase = phase_name

    def _lazy_init(self, num_labels: int) -> None:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self._model is None:
            if num_labels is None:
                raise ValueError("num_labels e necessario para inicializar o modelo.")
            self.num_labels = num_labels
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
            self._model.to(self.device)
        elif num_labels != self.num_labels:
            raise ValueError(
                f"num_labels mudou entre fases ({self.num_labels} -> {num_labels}). "
                "Isso indica inconsistencia no dataset."
            )
