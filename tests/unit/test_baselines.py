from bio_is_curriculum.baselines import REGISTRY, get_baseline


def test_baseline_registry():
    assert 1 in REGISTRY
    assert 2 in REGISTRY
    b1 = get_baseline(1)
    b2 = get_baseline(2)
    assert b1.INDEX == 1
    assert b1.REQUIRES_BIOIS is False
    assert b1.TRAINER_KIND == "phased"
    assert b2.INDEX == 2
    assert b2.TRAINER_KIND == "dynamic"
