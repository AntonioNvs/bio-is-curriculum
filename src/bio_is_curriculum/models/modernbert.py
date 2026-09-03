"""Backend ModernBERT para o curriculum learning.

Implementa `CurriculumModel` usando `answerdotai/ModernBERT-base` (ou qualquer
checkpoint do HuggingFace Hub compativel com `AutoModelForSequenceClassification`).

O loop de treino e feito manualmente (sem `Trainer`) para:
- suportar `sample_weight` por instancia;
- nao exigir `accelerate`;
- manter warm start real entre fases (o modelo nao e re-instanciado,
  apenas o optimizer/scheduler sao renovados a cada `fit_stage`).
"""
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from bio_is_curriculum.models.base import CurriculumModel
from bio_is_curriculum.curriculum.class_balance import balanced_epoch_weights
from bio_is_curriculum.curriculum.imbalance_losses import (
    ImbalanceLossConfig,
    class_weights_from_config,
    compute_loss_per_sample,
    maybe_logit_adjustment,
    validate_imbalance_method,
)
from bio_is_curriculum.data.augmentation import augment_minority_texts

_NO_DECAY_SUFFIXES = ("bias", "LayerNorm.weight", "norm.weight", "RMSNorm.weight")


def _seed_all(seed: int) -> None:
    """Fixa seeds e o backend determinístico do cuDNN/cuBLAS."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _classification_backbone(model):
    """Resolve encoder submodule across ModernBERT / RoBERTa / BERT heads."""
    for attr in ("model", "roberta", "bert"):
        backbone = getattr(model, attr, None)
        if backbone is not None:
            return backbone
    raise AttributeError(
        f"No known classification backbone on {type(model).__name__}"
    )


class _TextDataset(Dataset):
    """Guarda apenas os textos crus + labels/pesos; tokenização é per-batch."""

    def __init__(self, texts: list[str], labels: np.ndarray, weights: np.ndarray):
        self.texts = list(texts)
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx]),
            "weight": float(self.weights[idx]),
        }


class _DynamicPadCollator:
    """Tokeniza por batch para padding mínimo. Determinístico por construção."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc["weights"] = torch.tensor([b["weight"] for b in batch], dtype=torch.float)
        return enc


class ModernBertModel(CurriculumModel):
    """Modelo ModernBERT fine-tunado de forma faseada (warm start entre fases).

    Parameters
    ----------
    model_name : str
        Nome do checkpoint HuggingFace (default: ``"answerdotai/ModernBERT-base"``).
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        num_labels: int | None = None,
        epochs_per_stage: int = 2,
        batch_size: int = 32,
        eval_batch_size: int = 64,
        max_length: int = 256,
        lr: float = 2e-5,
        weight_decay: float = 1e-3,
        warmup_ratio: float = 0.06,
        imbalance_method: str = "inverse_freq_ce",
        class_balanced_loss: bool | None = None,
        effective_num_beta: float = 0.9999,
        dist_bal_tau: float = 1.0,
        dist_bal_logit_bias: float = 0.1,
        aug_target_min_count: int = 5,
        aug_ratio: float = 0.25,
        aug_random_swap: float = 0.2,
        aug_random_delete: float = 0.1,
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
        if class_balanced_loss is not None:
            imbalance_method = "inverse_freq_ce" if class_balanced_loss else "none"
        self.imbalance_method = validate_imbalance_method(imbalance_method)
        self.loss_cfg = ImbalanceLossConfig(
            method=self.imbalance_method,
            effective_num_beta=effective_num_beta,
            dist_bal_tau=dist_bal_tau,
            dist_bal_logit_bias=dist_bal_logit_bias,
        )
        self.aug_target_min_count = int(aug_target_min_count)
        self.aug_ratio = float(aug_ratio)
        self.aug_random_swap = float(aug_random_swap)
        self.aug_random_delete = float(aug_random_delete)
        self.random_state = random_state
        self.history_callback = history_callback

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _seed_all(self.random_state)

        self._tokenizer = None
        self._model = None
        self.global_step_: int = 0
        self._current_phase: str = "unknown"
        self._token_count_total: int = 0
        self._sample_count_total: int = 0
        self._best_val_macro_f1: float = float("nan")
        self._best_val_epoch: float = float("nan")
        self._steps_to_best_val: float = float("nan")
        self._fit_stage_calls: int = 0

    def fit_stage(
        self,
        texts: list[str],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
        balanced_sampling: bool = False,
    ):
        """Continua (ou inicia) o fine-tuning por `epochs_per_stage` epocas."""
        stage_seed = self.random_state + self._fit_stage_calls
        self._fit_stage_calls += 1
        _seed_all(stage_seed)

        y = np.asarray(y, dtype=np.int64)
        texts = list(texts)
        n = len(y)
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=np.float64)
        sample_weight = np.array(sample_weight, dtype=np.float64)

        if self.num_labels is not None:
            num_labels = int(self.num_labels)
        else:
            num_labels = int(np.max(y)) + 1
        self._lazy_init(num_labels)

        if self.imbalance_method == "minority_eda":
            texts, y, sample_weight, aug_stats = augment_minority_texts(
                texts=texts,
                y=y,
                sample_weight=sample_weight,
                target_min_count=self.aug_target_min_count,
                aug_ratio=self.aug_ratio,
                random_swap_prob=self.aug_random_swap,
                random_delete_prob=self.aug_random_delete,
                random_state=stage_seed,
            )
            print(
                f"[{self._current_phase}] minority_eda: "
                f"{aug_stats.n_before} -> {aug_stats.n_after} (+{aug_stats.n_added})"
            )

        n = len(y)
        class_weights_np = class_weights_from_config(y, num_labels, self.loss_cfg)
        class_weights = None
        if class_weights_np is not None:
            class_weights = torch.tensor(
                class_weights_np,
                dtype=torch.float,
                device=self.device,
            )
        logit_adjustment = maybe_logit_adjustment(
            y=y,
            num_labels=num_labels,
            cfg=self.loss_cfg,
            device=self.device,
        )

        dataset = _TextDataset(texts, y.astype(np.int64), sample_weight)
        collator = _DynamicPadCollator(self._tokenizer, self.max_length)
        shuffle_gen = torch.Generator()
        shuffle_gen.manual_seed(stage_seed)

        if balanced_sampling:
            sampler_weights, num_samples = balanced_epoch_weights(y)
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sampler_weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
                generator=shuffle_gen,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=collator,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collator,
                generator=shuffle_gen,
            )

        total_steps = len(loader) * self.epochs_per_stage
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        decay_params = []
        no_decay_params = []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.endswith(suffix) for suffix in _NO_DECAY_SUFFIXES):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
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

                loss_per_sample = compute_loss_per_sample(
                    logits,
                    labels,
                    class_weights=class_weights,
                    logit_adjustment=logit_adjustment,
                )
                loss = (loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-12)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                self.global_step_ += 1
                epoch_loss += loss.item()
                self._token_count_total += int(attention_mask.sum().item())
                self._sample_count_total += int(attention_mask.shape[0])

                if self.history_callback is not None:
                    self.history_callback({
                        "event": "train_step",
                        "phase": self._current_phase,
                        "epoch": epoch + 1,
                        "step": self.global_step_,
                        "loss": round(loss.item(), 6),
                        "lr": scheduler.get_last_lr()[0],
                        "avg_seq_len": self._avg_seq_len(),
                        "compute_proxy": self._compute_proxy(),
                    })

            if X_val is not None and y_val is not None and len(y_val) > 0:
                val_proba = self.predict_proba(X_val)
                val_preds = np.argmax(val_proba, axis=1)
                val_macro = float(f1_score(y_val, val_preds, average="macro"))
                val_micro = float(f1_score(y_val, val_preds, average="micro"))
                val_weighted = float(f1_score(y_val, val_preds, average="weighted"))
                val_acc = float(accuracy_score(y_val, val_preds))

                if np.isnan(self._best_val_macro_f1) or val_macro > self._best_val_macro_f1:
                    self._best_val_macro_f1 = val_macro
                    self._best_val_epoch = float(epoch + 1)
                    self._steps_to_best_val = float(self.global_step_)

                if self.history_callback is not None:
                    self.history_callback({
                        "event": "epoch_end",
                        "phase": self._current_phase,
                        "epoch": epoch + 1,
                        "step": self.global_step_,
                        "loss": round(epoch_loss / max(len(loader), 1), 6),
                        "lr": scheduler.get_last_lr()[0],
                        "val_macro_f1": val_macro,
                        "val_micro_f1": val_micro,
                        "val_f1_weighted": val_weighted,
                        "val_accuracy": val_acc,
                        "avg_seq_len": self._avg_seq_len(),
                        "compute_proxy": self._compute_proxy(),
                    })

        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self._lazy_init(self.num_labels)
        self._model.eval()

        dataset = _TextDataset(
            list(texts),
            np.zeros(len(texts), dtype=np.int64),
            np.ones(len(texts), dtype=np.float64),
        )
        collator = _DynamicPadCollator(self._tokenizer, self.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

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

    def get_training_stats(self) -> dict:
        return {
            "avg_seq_len": self._avg_seq_len(),
            "compute_proxy": self._compute_proxy(),
            "best_val_macro_f1": self._best_val_macro_f1,
            "best_val_epoch": self._best_val_epoch,
            "steps_to_best_val": self._steps_to_best_val,
        }

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

    def _avg_seq_len(self) -> float:
        if self._sample_count_total == 0:
            return float("nan")
        return float(self._token_count_total / self._sample_count_total)

    def _compute_proxy(self) -> float:
        avg_seq_len = self._avg_seq_len()
        if np.isnan(avg_seq_len):
            return float("nan")
        return float(self.global_step_ * avg_seq_len)

    def extract_hidden_states(self, texts: list[str]) -> list[np.ndarray]:
        """Last-layer token hidden states per sample for nuclear-norm scoring."""
        self._lazy_init(self.num_labels)
        self._model.eval()
        backbone = _classification_backbone(self._model)

        dataset = _TextDataset(
            list(texts),
            np.zeros(len(texts), dtype=np.int64),
            np.ones(len(texts), dtype=np.float64),
        )
        collator = _DynamicPadCollator(self._tokenizer, self.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        all_hidden: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden = outputs.last_hidden_state.cpu().numpy()
                for i in range(hidden.shape[0]):
                    seq_len = int(attention_mask[i].sum().item())
                    all_hidden.append(hidden[i, :seq_len, :])
        return all_hidden

    def fit_epoch(
        self,
        texts: list[str],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val=None,
        y_val=None,
        epoch_seed: int | None = None,
    ) -> None:
        """Train exactly one epoch (used by dynamic curricula)."""
        prev_epochs = self.epochs_per_stage
        self.epochs_per_stage = 1
        if epoch_seed is not None:
            _seed_all(epoch_seed)
        self.fit_stage(texts, y, sample_weight=sample_weight, X_val=X_val, y_val=y_val)
        self.epochs_per_stage = prev_epochs
