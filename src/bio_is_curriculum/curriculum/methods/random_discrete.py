"""Random-phase discrete curriculum: BIOIS selection with a random difficulty order.

Control condition for the difficulty-signal ablation. It answers "would random
phases do as well as the BIO-IS difficulty ordering?" while holding everything
else fixed: the redundancy scores ``r``, the noise term, the hard-phase
weighting ``1 - beta * min(r, r_cap)``, the noise downweighting, and the
class-coverage tie-break are all inherited unchanged from
``BIOISDiscreteCurriculum``. Only the per-instance difficulty score ``e`` — the
sole input to the per-class phase masks — is replaced by a uniform draw.

Because ``per_class_low_quantile_mask`` re-quantiles inside its per-class loop,
a *global* uniform ``e`` still yields the same per-class phase quotas as BIOIS:
the stratification is enforced by the mask, not by the score. Classes with
``<= RARE_CLASS_PIN_THRESHOLD`` members remain pinned into every phase.
"""
from __future__ import annotations

import numpy as np

from bio_is_curriculum.curriculum.methods.biois_discrete import BIOISDiscreteCurriculum
from bio_is_curriculum.signals.biois import extract_biois_curriculum_signals


class RandomDiscreteCurriculum(BIOISDiscreteCurriculum):
    """Discrete curriculum whose phase ordering is random, not difficulty-driven."""

    # Inherited as True because ``r`` and the noise term still come from BIOIS.
    REQUIRES_BIOIS = True
    METHOD_ID = "random_discrete"
    PHASE_NAMES = ("clean", "diverse", "hard")

    # The BIOIS difficulty vector, kept so the class-coverage tie-break can stay
    # identical to BIOIS instead of following the random ordering. Set in
    # ``_extract_signals``; declared here to keep the inherited ``__init__``
    # signature (and therefore ``BaseEstimator.get_params``) untouched.
    _e_biois: np.ndarray | None = None

    def _extract_signals(self, selector, y):
        r, e_biois, noise = extract_biois_curriculum_signals(
            selector,
            y,
            self._texts_build,
            margin_weight=self.margin_weight,
            entropy_weight=self.entropy_weight,
            length_weight=self.length_weight,
        )
        self._noise_ = noise
        self._e_biois = e_biois
        # The score is computed once, up front, so a single draw is enough and
        # the result is deterministic under ``random_state``.
        rng = np.random.default_rng(self.random_state)
        return r, rng.random(len(e_biois))

    def _coverage_score_from_signals(self, r: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Keep the BIOIS coverage ordering rather than the random one.

        ``e`` feeds both the phase masks and the class-coverage tie-break in
        ``balance_phase_indices``. Returning the random ``e`` here would make
        the control "random phases *and* random coverage", so the BIOIS vector
        is used instead and the ordering of phases is the only variable.
        """
        if self._e_biois is None:
            return super()._coverage_score_from_signals(r, e)
        return -np.asarray(self._e_biois, dtype=np.float64)
