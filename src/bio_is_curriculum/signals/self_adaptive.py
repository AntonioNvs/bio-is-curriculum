"""Self-adaptive PLM difficulty (Feng, Liu & Schütze, ACL SRW 2025).

Adaptation for this codebase:
- Score with a frozen ``AutoModelForMaskedLM`` (no parameter updates).
- Use a generic cloze prompt plus one-token verbalizers.
- Because repository labels are numeric, verbalizers are derived automatically
  from class-contrastive TF-IDF on the fitting split (not hand-crafted).
- Multiclass confidence = top-1 minus top-2 verbalizer probability after
  normalizing over class tokens (paper §3.3). Higher confidence = easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_PROMPT_SUFFIX = " This text is [MASK]."


@dataclass(frozen=True)
class VerbalizerSelection:
    """One-token verbalizer mapping for a labeled fold split."""

    tokens: list[str]
    token_ids: list[int]
    scores: list[float]
    class_ids: list[int]


def _is_single_token(tokenizer, word: str) -> int | None:
    """Return vocab id if ``word`` is exactly one non-special token, else None."""
    word = word.strip()
    if not word or any(ch.isspace() for ch in word):
        return None
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        return None
    tid = int(ids[0])
    special = set(tokenizer.all_special_ids)
    if tid in special:
        return None
    tok = tokenizer.convert_ids_to_tokens(tid)
    if tok is None:
        return None
    # Reject pure punctuation / control-like pieces.
    stripped = tok.lstrip("Ġ▁#")
    if not stripped or not any(c.isalnum() for c in stripped):
        return None
    return tid


def select_verbalizers(
    texts: list[str],
    y: np.ndarray,
    tokenizer,
    *,
    min_df: int = 1,
    max_features: int = 50000,
) -> VerbalizerSelection:
    """Pick a unique single-token verbalizer per class via contrastive TF-IDF."""
    y_arr = np.asarray(y, dtype=np.int64)
    class_ids = sorted(int(c) for c in np.unique(y_arr))
    if len(class_ids) < 2:
        raise ValueError("Need at least two classes for verbalizer selection.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    n_docs = X.shape[0]

    # Pre-filter vocabulary to single tokenizer tokens.
    eligible: list[tuple[str, int]] = []
    for word in vocab:
        tid = _is_single_token(tokenizer, word)
        if tid is not None:
            eligible.append((str(word), tid))
    if len(eligible) < len(class_ids):
        # Fallback: scan tokenizer vocab for alphanumeric single tokens.
        for tid in range(min(len(tokenizer), 50000)):
            if tid in set(tokenizer.all_special_ids):
                continue
            tok = tokenizer.convert_ids_to_tokens(tid)
            if tok is None:
                continue
            word = tok.lstrip("Ġ▁#")
            if word.isalpha() and len(word) >= 2:
                eligible.append((word.lower(), tid))
            if len(eligible) >= max(2000, len(class_ids) * 20):
                break

    if len(eligible) < len(class_ids):
        raise RuntimeError(
            f"Only {len(eligible)} eligible verbalizer tokens for "
            f"{len(class_ids)} classes."
        )

    # Build word -> column index for TF-IDF features when available.
    word_to_col = {w: i for i, w in enumerate(vocab)}
    candidates: dict[int, list[tuple[float, str, int]]] = {c: [] for c in class_ids}

    for cls in class_ids:
        mask = y_arr == cls
        if not np.any(mask):
            continue
        mean_pos = np.asarray(X[mask].mean(axis=0)).ravel()
        mean_neg = (
            np.asarray(X[~mask].mean(axis=0)).ravel()
            if np.any(~mask)
            else np.zeros_like(mean_pos)
        )
        contrast = mean_pos - mean_neg
        scored: list[tuple[float, str, int]] = []
        for word, tid in eligible:
            col = word_to_col.get(word)
            score = float(contrast[col]) if col is not None else 0.0
            # Mild prior for longer content words when TF-IDF is unavailable.
            if col is None:
                score = float(len(word)) / 20.0
            scored.append((score, word, tid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        candidates[cls] = scored

    # Greedy unique assignment by best available contrastive score.
    chosen_tokens: dict[int, str] = {}
    chosen_ids: dict[int, int] = {}
    chosen_scores: dict[int, float] = {}
    used_ids: set[int] = set()
    used_tokens: set[str] = set()

    # Priority: classes with fewer high-scoring options first? Global greedy:
    # repeatedly pick the globally best unused (class, token) pair.
    pool: list[tuple[float, int, str, int]] = []
    for cls, scored in candidates.items():
        for score, word, tid in scored[: max(200, len(class_ids) * 5)]:
            pool.append((score, cls, word, tid))
    pool.sort(key=lambda t: (-t[0], t[1], t[2]))

    for score, cls, word, tid in pool:
        if cls in chosen_tokens:
            continue
        if tid in used_ids or word in used_tokens:
            continue
        chosen_tokens[cls] = word
        chosen_ids[cls] = tid
        chosen_scores[cls] = float(score)
        used_ids.add(tid)
        used_tokens.add(word)
        if len(chosen_tokens) == len(class_ids):
            break

    missing = [c for c in class_ids if c not in chosen_tokens]
    if missing:
        raise RuntimeError(
            f"Could not assign unique verbalizers for classes {missing} "
            f"(n_docs={n_docs}, eligible={len(eligible)})."
        )

    return VerbalizerSelection(
        tokens=[chosen_tokens[c] for c in class_ids],
        token_ids=[chosen_ids[c] for c in class_ids],
        scores=[chosen_scores[c] for c in class_ids],
        class_ids=class_ids,
    )


def build_prompted_texts(
    texts: list[str],
    prompt_suffix: str,
    mask_token: str,
) -> list[str]:
    """Append a cloze template, substituting ``[MASK]`` with the model mask token."""
    suffix = prompt_suffix.replace("[MASK]", mask_token)
    if not suffix.startswith(" "):
        suffix = " " + suffix
    return [f"{text.rstrip()}{suffix}" for text in texts]


def confidence_from_class_probs(class_probs: np.ndarray) -> np.ndarray:
    """Paper §3.3: |P_max - P_second-max| (binary reduces to |P_pos - P_neg|)."""
    probs = np.asarray(class_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("class_probs must be (n_samples, n_classes) with K>=2")
    # Sort descending along classes.
    part = np.partition(probs, -2, axis=1)
    top2 = part[:, -1]
    second = part[:, -2]
    return np.clip(top2 - second, 0.0, 1.0)


def difficulty_from_confidence(confidence: np.ndarray) -> np.ndarray:
    """Convert paper confidence (higher=easier) to curriculum difficulty (higher=harder)."""
    return 1.0 - np.asarray(confidence, dtype=np.float64)


@torch.no_grad()
def score_mlm_confidence(
    texts: list[str],
    verbalizer: VerbalizerSelection,
    *,
    model_name: str,
    prompt_suffix: str = DEFAULT_PROMPT_SUFFIX,
    max_length: int = 256,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Frozen MLM forward pass → per-example confidence in [0, 1]."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.mask_token is None:
        raise ValueError(f"Tokenizer for {model_name!r} has no mask_token.")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    prompted = build_prompted_texts(texts, prompt_suffix, tokenizer.mask_token)
    token_ids = torch.tensor(verbalizer.token_ids, dtype=torch.long, device=device)
    confidences = np.zeros(len(prompted), dtype=np.float64)

    # Reserve room for the prompt suffix when truncating the document.
    suffix_ids = tokenizer.encode(
        prompt_suffix.replace("[MASK]", tokenizer.mask_token),
        add_special_tokens=False,
    )
    # Leave headroom for special tokens.
    doc_budget = max(8, max_length - len(suffix_ids) - 4)

    for start in range(0, len(prompted), batch_size):
        batch_texts = prompted[start : start + batch_size]
        # Truncate long docs before the suffix by rebuilding from originals.
        trimmed = []
        for i, raw in enumerate(texts[start : start + batch_size]):
            doc_ids = tokenizer.encode(raw, add_special_tokens=False)
            if len(doc_ids) > doc_budget:
                raw = tokenizer.decode(doc_ids[:doc_budget], skip_special_tokens=True)
            trimmed.append(
                build_prompted_texts([raw], prompt_suffix, tokenizer.mask_token)[0]
            )
        enc = tokenizer(
            trimmed,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        mask_id = tokenizer.mask_token_id
        # Last MASK occurrence per row (prompt is appended).
        is_mask = input_ids == mask_id
        if not bool(is_mask.any()):
            raise RuntimeError("No mask token found in prompted batch.")
        # For rows missing MASK after truncation, fall back to last non-pad.
        mask_pos = torch.full(
            (input_ids.shape[0],), -1, dtype=torch.long, device=device
        )
        for row in range(input_ids.shape[0]):
            positions = torch.nonzero(is_mask[row], as_tuple=False).flatten()
            if len(positions) == 0:
                raise RuntimeError(
                    f"Example {start + row} lost its mask token after truncation."
                )
            mask_pos[row] = positions[-1]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        gather_index = mask_pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        mask_logits = torch.gather(logits, 1, gather_index).squeeze(1)  # (B, V)
        class_logits = mask_logits.index_select(1, token_ids)  # (B, K)
        class_probs = torch.softmax(class_logits, dim=-1).cpu().numpy()
        confidences[start : start + len(batch_texts)] = confidence_from_class_probs(
            class_probs
        )

    meta = {
        "model_name": model_name,
        "prompt_suffix": prompt_suffix,
        "max_length": max_length,
        "n_examples": len(texts),
        "verbalizer_tokens": list(verbalizer.tokens),
        "verbalizer_token_ids": list(verbalizer.token_ids),
        "verbalizer_scores": list(verbalizer.scores),
        "verbalizer_class_ids": list(verbalizer.class_ids),
        "confidence_mean": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
    }
    # Free MLM weights before classifier training.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return confidences, meta


def pmd_rank_weights(n: int, exponent: float = 2.0) -> np.ndarray:
    """Paper Eq. for P(x_n) ∝ n^exponent over ranks 1..N."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    w = np.power(ranks, float(exponent))
    return w / w.sum()


def pmd_partition_counts(batch_size: int, hard_fraction: float = 0.6) -> tuple[int, int]:
    """Return (n_hard, n_easy) with hard partition prioritized (paper 6:4)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    hard_fraction = float(hard_fraction)
    if not 0.0 < hard_fraction < 1.0 and batch_size > 1:
        # Allow 1.0 / 0.0 for degenerate unit tests, but prefer interior.
        hard_fraction = min(max(hard_fraction, 0.0), 1.0)
    n_hard = int(round(batch_size * hard_fraction))
    n_hard = min(max(n_hard, 0), batch_size)
    n_easy = batch_size - n_hard
    if batch_size >= 2:
        if n_hard == 0:
            n_hard, n_easy = 1, batch_size - 1
        elif n_easy == 0:
            n_hard, n_easy = batch_size - 1, 1
    return n_hard, n_easy


def pmd_sampling_probs(
    confidence: np.ndarray,
    *,
    rank_exponent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build hard/easy multinomial probs over original indices (PMD, paper §3.4.3).

    Confidence is sorted descending (easy→hard). Higher rank indices get higher
    hard-partition probability; the reverse feeds the easy partition.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    n = len(conf)
    order = np.argsort(-conf, kind="mergesort")  # stable: easy → hard
    rank_w = pmd_rank_weights(n, exponent=rank_exponent)
    hard_probs = np.zeros(n, dtype=np.float64)
    easy_probs = np.zeros(n, dtype=np.float64)
    hard_probs[order] = rank_w
    easy_probs[order] = rank_w[::-1]
    return hard_probs, easy_probs, order
