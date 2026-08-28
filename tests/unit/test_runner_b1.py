"""Runner wiring tests for b1 (no BIOIS on pure b1 mode)."""

from bio_is_curriculum.baselines import get_baseline
from bio_is_curriculum.baselines.base import DynamicBaselineBase
from bio_is_curriculum.pipeline.modes import parse_is_baseline_index


def test_b1_registry_skips_biois_gate():
    baseline_cls = get_baseline(1)
    needs_biois_for_baseline = (
        baseline_cls is not None
        and not issubclass(baseline_cls, DynamicBaselineBase)
        and getattr(baseline_cls, "REQUIRES_BIOIS", True)
    )
    assert needs_biois_for_baseline is False


def test_is_b1_still_needs_biois_for_instance_selection():
    is_baseline_idx = parse_is_baseline_index("is_b1")
    assert is_baseline_idx == 1
    needs_is = is_baseline_idx is not None
    assert needs_is is True
