from bio_is_curriculum.pipeline.modes import (
    parse_is_baseline_index,
    uses_is_subset,
)


def test_is_b2_mode_parsing():
    assert parse_is_baseline_index("is_b2") == 2
    assert parse_is_baseline_index("b2") is None
    assert uses_is_subset("is_b2")
    assert not uses_is_subset("b2")
