"""Registry de metodos de curriculum learning."""
from __future__ import annotations

from typing import Any

from curriculum.methods.biois_discrete import BIOISDiscreteCurriculum
from curriculum.methods.spcl_loss import SPCLLossCurriculum
from curriculum.methods.spcl_soft import SPCLSoftCurriculum

REGISTRY: dict[str, type] = {
    BIOISDiscreteCurriculum.METHOD_ID: BIOISDiscreteCurriculum,
    SPCLSoftCurriculum.METHOD_ID: SPCLSoftCurriculum,
    SPCLLossCurriculum.METHOD_ID: SPCLLossCurriculum,
}

# Aliases de conveniencia.
ALIASES: dict[str, str] = {
    "discrete": "biois_discrete",
    "biois": "biois_discrete",
    "continuous": "spcl_soft",
    "spcl": "spcl_soft",
    "loss": "spcl_loss",
}


def resolve_method_id(method: str) -> str:
    """Normaliza ID ou alias para o identificador canonico."""
    key = method.strip().lower()
    if key in REGISTRY:
        return key
    if key in ALIASES:
        return ALIASES[key]
    disponiveis = sorted(REGISTRY) + sorted(ALIASES)
    raise ValueError(
        f"Metodo de curriculum {method!r} nao encontrado. "
        f"Disponiveis: {disponiveis}"
    )


def get_curriculum_method(method: str) -> type:
    """Retorna a classe do metodo de curriculum."""
    return REGISTRY[resolve_method_id(method)]


def build_curriculum_kwargs(method: str, args) -> dict[str, Any]:
    """Monta kwargs do construtor a partir dos argumentos CLI."""
    method_id = resolve_method_id(method)
    common = {
        "beta": args.curriculum_beta,
        "hard_slice_quantile": args.hard_slice_quantile,
        "random_state": args.random_state,
    }
    if method_id == "biois_discrete":
        q_low, q_mid, q_high = args.curriculum_q
        return {
            **common,
            "q_low": q_low,
            "q_mid": q_mid,
            "q_high": q_high,
        }
    if method_id == "spcl_soft":
        return {
            **common,
            "n_steps": args.curriculum_n_steps,
            "alpha_decay": args.curriculum_alpha_decay,
            "lambda_init": args.curriculum_soft_lambda_init,
            "lambda_growth": args.curriculum_soft_lambda_growth,
            "lambda_max": args.curriculum_soft_lambda_max,
            "min_weight": args.curriculum_soft_min_weight,
            "stability_tol": args.curriculum_soft_stability_tol,
            "saturation_patience": args.curriculum_soft_saturation_patience,
            "max_effective_steps": args.curriculum_soft_max_effective_steps,
        }
    if method_id == "spcl_loss":
        return {
            **common,
            "scheme": getattr(args, "curriculum_loss_scheme", "linear"),
            "n_steps": args.curriculum_n_steps,
            "lambda_init": args.curriculum_lambda_init,
            "lambda_mult": args.curriculum_lambda_mult,
            "lambda_step": getattr(args, "curriculum_lambda_step", 0.5),
            "lambda_max": getattr(args, "curriculum_lambda_max", None),
            "lambda2": getattr(args, "curriculum_lambda2", None),
            "prior_use_reliability": getattr(
                args, "curriculum_loss_prior_reliability", True
            ),
            "min_weight": args.curriculum_min_weight,
        }
    raise ValueError(f"Metodo sem factory de kwargs: {method_id}")
