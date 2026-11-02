"""Paired statistical tests over cross-validation folds.

For every paper table and dataset, resolves the latest run per method
(``discovery.identify_latest``), loads per-fold ``macro_f1`` (or another metric)
and compares all method pairs with a paired Student t-test and a Wilcoxon
signed-rank test. p-values are adjusted within each (table, dataset) family via
Holm-Bonferroni.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from bio_is_curriculum.results.aggregator import collect_per_fold_map
from bio_is_curriculum.results.discovery import identify_latest
from bio_is_curriculum.results.paper_tables import PAPER_TABLES, TableSpec


def holm_adjust(p_values, alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (returned in input order).

    NaN p-values are treated as 1.0 (conservative: never significant).
    """
    p = np.asarray(p_values, dtype=float)
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    adjusted_sorted = np.empty(m, dtype=float)
    for k in range(m):  # k-th smallest (0-indexed)
        adjusted_sorted[k] = min(1.0, (m - k) * sorted_p[k])
    for k in range(1, m):
        adjusted_sorted[k] = max(adjusted_sorted[k], adjusted_sorted[k - 1])
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    if np.all(d == 0):
        return 1.0
    try:
        res = stats.wilcoxon(
            d, alternative="two-sided", method="auto", zero_method="wilcox"
        )
        p = float(res.pvalue)
        return p if not np.isnan(p) else float("nan")
    except Exception:
        return float("nan")


def paired_tests(
    values: dict[str, dict[int, float]],
    *,
    table: str,
    dataset: str,
    alpha: float = 0.05,
    pairs: set[frozenset[str]] | None = None,
) -> list[dict]:
    """Run pairwise paired tests over aligned folds and Holm-adjust.

    ``pairs`` optionally restricts the comparisons to the given (unordered)
    method-name pairs; ``None`` tests every pair.
    """
    methods = sorted(values.keys())
    rows: list[dict] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            name_a, name_b = methods[i], methods[j]
            if pairs is not None and frozenset({name_a, name_b}) not in pairs:
                continue
            folds = sorted(set(values[name_a]) & set(values[name_b]))
            a = np.array([values[name_a][f] for f in folds], dtype=float)
            b = np.array([values[name_b][f] for f in folds], dtype=float)
            if len(folds) < 2:
                rows.append(
                    {
                        "table": table,
                        "dataset": dataset,
                        "method_a": name_a,
                        "method_b": name_b,
                        "n_folds": len(folds),
                        "mean_a": float(np.mean(a)) if len(a) else float("nan"),
                        "mean_b": float(np.mean(b)) if len(b) else float("nan"),
                        "delta": float("nan"),
                        "t_stat": float("nan"),
                        "p_ttest": float("nan"),
                        "p_wilcoxon": float("nan"),
                    }
                )
                continue
            t_stat, p_ttest = stats.ttest_rel(a, b)
            rows.append(
                {
                    "table": table,
                    "dataset": dataset,
                    "method_a": name_a,
                    "method_b": name_b,
                    "n_folds": len(folds),
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "delta": float(np.mean(a) - np.mean(b)),
                    "t_stat": float(t_stat),
                    "p_ttest": float(p_ttest),
                    "p_wilcoxon": _wilcoxon_p(a, b),
                }
            )

    p_ttest_holm = holm_adjust([r["p_ttest"] for r in rows], alpha)
    p_wilcox_holm = holm_adjust([r["p_wilcoxon"] for r in rows], alpha)
    for r, pth, pwh in zip(rows, p_ttest_holm, p_wilcox_holm):
        r["p_ttest_holm"] = float(pth)
        r["sig_ttest_holm"] = bool(pth < alpha)
        r["p_wilcoxon_holm"] = float(pwh)
        r["sig_wilcoxon_holm"] = bool(pwh < alpha)
    return rows


def run(
    results_dir: str = "results",
    metric: str = "macro_f1",
    n_splits: int = 5,
    alpha: float = 0.05,
    tables: list[TableSpec] | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Resolve latest runs and run paired tests across the paper tables.

    ``pairs`` optionally restricts comparisons to the given method-name pairs
    (e.g. ``[("Raw", "IS+CL")]``); ``None`` tests every pair.
    """
    tables = tables or PAPER_TABLES
    pair_set = {frozenset(p) for p in pairs} if pairs is not None else None
    mapping = identify_latest(results_dir, tables=tables, n_splits=n_splits)

    all_rows: list[dict] = []
    for table in tables:
        for dataset in table.datasets:
            sub = mapping[
                (mapping["table"] == table.name) & (mapping["dataset"] == dataset)
            ]
            values: dict[str, dict[int, float]] = {}
            for _, row in sub.iterrows():
                if not row["path"]:
                    continue
                values[str(row["method"])] = collect_per_fold_map(
                    str(row["path"]),
                    str(row["mode"]),
                    list(range(n_splits)),
                    metric,
                )
            if len(values) < 2:
                continue
            all_rows.extend(
                paired_tests(
                    values,
                    table=table.name,
                    dataset=dataset,
                    alpha=alpha,
                    pairs=pair_set,
                )
            )

    cols = [
        "table",
        "dataset",
        "method_a",
        "method_b",
        "n_folds",
        "mean_a",
        "mean_b",
        "delta",
        "t_stat",
        "p_ttest",
        "p_ttest_holm",
        "sig_ttest_holm",
        "p_wilcoxon",
        "p_wilcoxon_holm",
        "sig_wilcoxon_holm",
    ]
    return pd.DataFrame(all_rows, columns=cols)


def _print_report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    for (table, dataset), group in df.groupby(["table", "dataset"], sort=False):
        print(f"\n=== {table} / {dataset} ===")
        view = group[
            [
                "method_a",
                "method_b",
                "n_folds",
                "mean_a",
                "mean_b",
                "delta",
                "p_ttest_holm",
                "sig_ttest_holm",
                "p_wilcoxon_holm",
                "sig_wilcoxon_holm",
            ]
        ].copy()
        print(view.to_string(index=False))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Paired significance tests (t-test + Wilcoxon, Holm-adjusted) over "
            "CV folds for the paper's result tables."
        )
    )
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    parser.add_argument("--metric", default="macro_f1", help="Metric to test.")
    parser.add_argument("--n-splits", type=int, default=5, help="Expected CV folds.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--pairs",
        action="append",
        default=None,
        metavar="A,B",
        help=(
            "Restrict to one method pair (comma-separated, e.g. 'Raw,IS+CL'). "
            "Repeat for multiple pairs. Default: all pairs."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional CSV path for results.")
    return parser.parse_args(argv)


def _parse_pairs(specs: list[str] | None) -> list[tuple[str, str]] | None:
    if not specs:
        return None
    pairs: list[tuple[str, str]] = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid --pairs value {spec!r}; expected 'A,B'.")
        pairs.append((parts[0], parts[1]))
    return pairs


def main(argv=None) -> None:
    args = _parse_args(argv)
    df = run(
        results_dir=args.results_dir,
        metric=args.metric,
        n_splits=args.n_splits,
        alpha=args.alpha,
        pairs=_parse_pairs(args.pairs),
    )
    if args.output:
        df.to_csv(args.output, index=False, float_format="%.6f")
        print(f"Saved significance results to: {args.output}")
    _print_report(df)


if __name__ == "__main__":
    main()
