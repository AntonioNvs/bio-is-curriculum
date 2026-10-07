"""Training-dynamics difficulty signals (Christopoulou et al., EMNLP 2022)."""

from __future__ import annotations

import numpy as np


def gold_label_probas(probas: np.ndarray, y, model=None) -> np.ndarray:
    """Per-sample P(y_i) from a probability matrix."""
    y_arr = np.asarray(y).astype(int)
    eps = 1e-12
    if probas.shape[1] > int(np.max(y_arr)):
        p_true = probas[np.arange(len(y_arr)), y_arr]
    elif model is not None and hasattr(model, "_clf") and hasattr(model._clf, "classes_"):
        cls = np.asarray(model._clf.classes_).astype(int)
        col_by_class = {c: i for i, c in enumerate(cls.tolist())}
        cols = np.array([col_by_class.get(int(lbl), -1) for lbl in y_arr], dtype=int)
        p_true = np.full(len(y_arr), eps, dtype=np.float64)
        valid = cols >= 0
        if np.any(valid):
            p_true[valid] = probas[np.arange(len(y_arr))[valid], cols[valid]]
    else:
        p_true = np.full(len(y_arr), eps, dtype=np.float64)
    return np.clip(p_true, eps, 1.0)


def confidence_difficulty(proba_trace: np.ndarray) -> np.ndarray:
    """Inverse normalized confidence: low avg P(y) => high difficulty."""
    trace = np.asarray(proba_trace, dtype=np.float64)
    if trace.ndim == 1:
        trace = trace[:, np.newaxis]
    mu = trace.mean(axis=1)
    lo, hi = float(mu.min()), float(mu.max())
    if hi > lo:
        mu_norm = (mu - lo) / (hi - lo)
    else:
        return np.zeros_like(mu)
    return 1.0 - mu_norm


def variability_difficulty(proba_trace: np.ndarray) -> np.ndarray:
    """Normalized std-dev of gold-label probas (Eq. 3 in Christopoulou et al.)."""
    trace = np.asarray(proba_trace, dtype=np.float64)
    if trace.ndim == 1:
        trace = trace[:, np.newaxis]
    var = trace.std(axis=1)
    lo, hi = float(var.min()), float(var.max())
    if hi > lo:
        return (var - lo) / (hi - lo)
    return np.zeros_like(var)


def clone_probe_model(model):
    """Return a fresh untrained copy of a ModernBERT model for TD probing."""
    from bio_is_curriculum.models.modernbert import ModernBertModel

    if not isinstance(model, ModernBertModel):
        raise ValueError("td_discrete requires ModernBERT backend.")
    return ModernBertModel(
        model_name=model.model_name,
        num_labels=model.num_labels,
        epochs_per_stage=1,
        batch_size=model.batch_size,
        eval_batch_size=model.eval_batch_size,
        max_length=model.max_length,
        lr=model.lr,
        weight_decay=model.weight_decay,
        warmup_ratio=model.warmup_ratio,
        imbalance_method=model.imbalance_method,
        effective_num_beta=model.loss_cfg.effective_num_beta,
        dist_bal_tau=model.loss_cfg.dist_bal_tau,
        dist_bal_logit_bias=model.loss_cfg.dist_bal_logit_bias,
        aug_target_min_count=model.aug_target_min_count,
        aug_ratio=model.aug_ratio,
        aug_random_swap=model.aug_random_swap,
        aug_random_delete=model.aug_random_delete,
        random_state=model.random_state,
    )


def collect_gold_label_probas(
    model,
    texts: list[str],
    y,
    *,
    n_epochs: int,
) -> np.ndarray:
    """Train a probe model for ``n_epochs`` and record P(y_i) after each epoch."""
    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")
    y_arr = np.asarray(y)
    traces: list[np.ndarray] = []
    for _ in range(n_epochs):
        model.fit_stage(list(texts), y_arr)
        probas = model.predict_proba(list(texts))
        traces.append(gold_label_probas(probas, y_arr, model=model))
    return np.column_stack(traces)


def training_dynamics_difficulty(
    model,
    texts: list[str],
    y,
    *,
    n_epochs: int = 2,
    metric: str = "confidence",
) -> np.ndarray:
    """Collect training dynamics on a probe model and return difficulty scores."""
    probe = clone_probe_model(model)
    trace = collect_gold_label_probas(probe, texts, y, n_epochs=n_epochs)
    if metric == "confidence":
        return confidence_difficulty(trace)
    if metric == "variability":
        return variability_difficulty(trace)
    raise ValueError(f"Unknown training-dynamics metric: {metric!r}")
