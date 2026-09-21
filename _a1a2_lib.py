# ==========================================================================
# Workstream 5 (A1/A2 ablations): shared library.
#
# Reuses the extracted-and-verified tabpfn_cf5.ipynb pipeline from _e5_lib.py
# (prepare_dataset, fit_models, proba_fn, stable_int_seed, l0_mixed -- all
# byte-identical to what run_benchmark uses, so factuals sampled here overlap
# with the main benchmark's evaluated instances for the same seed).
#
# Adds two PACE-only wrappers that extend completness.ipynb's "Section 5
# relaxation harness" (guided_cf_build_cache / pace_with_params) with the two
# knobs A2 needs that harness didn't expose yet:
#   - gamma (Stage A importance-sharpening exponent). stageA_build already
#     accepts it; the harness just never passed it through. gamma=0 is also
#     reused as the A2 "surrogate=none/uniform importance" condition, since
#     raw_unit ** 0 == 1 flattens Stage A's per-unit weights to uniform
#     without touching anchor/boundary pool selection (still driven by
#     predict_proba_guidance's p_train).
#   - order ("l0_l2" / "l2_l0"), forwarded to stageC_select_best's new
#     optional parameter (pfn_cf_guided_onehot.py) added for the A2
#     lexicographic-order ablation. Default "l0_l2" reproduces the paper's
#     unparameterized behaviour exactly.
#
# frac_anchor_mix / frac_boundary_mix / frac_guided_noise (pi_a/pi_b/pi_n, for
# A1) were already exposed by stageB_generate and already renormalize when
# one is zeroed (pfn_cf_guided_onehot.py:606-610) -- no wrapper change needed
# for those beyond passing them through, which pace_with_params in
# completness.ipynb already did; reproduced here for a single source of truth.
# ==========================================================================

from _e5_lib import *  # noqa: F401,F403  (prepare_dataset, fit_models, proba_fn,
                        # stable_int_seed, l0_mixed, DICE_CLIP_QUANTILES, and
                        # pfn_cf_guided_onehot's stageA_build/stageA_target_anchors_from_p/
                        # stageB_generate/stageC_select_best/FeatureInfo, all via *)

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class GuidedCFCacheAblation:
    X_tr_sc: np.ndarray
    predict_proba_guidance: Any
    feature_info: Optional[FeatureInfo]
    p_train: np.ndarray
    A: Any
    anchors_by_class: Dict[int, np.ndarray]
    anchors_diag_by_class: Dict[int, Dict[str, Any]]
    gamma: float


def guided_cf_build_cache_ablation(
    *,
    X_tr_sc,
    predict_proba_target,
    predict_proba_guidance=None,
    stageA_random_state: int = 0,
    anchor_k: int = 250,
    anchors_random_state: int = 1,
    feature_info=None,
    gamma: float = 1.0,
) -> GuidedCFCacheAblation:
    """Same as completness.ipynb's guided_cf_build_cache, plus gamma passthrough
    into stageA_build (A2's importance-sharpening / surrogate=uniform axis)."""
    if predict_proba_guidance is None:
        predict_proba_guidance = predict_proba_target

    p_train = predict_proba_guidance(X_tr_sc).astype(np.float64)
    A = stageA_build(
        X_train=X_tr_sc,
        predict_proba_fn=predict_proba_guidance,
        random_state=stageA_random_state,
        feature_info=feature_info,
        gamma=gamma,
    )

    anchors_by_class, anchors_diag_by_class = {}, {}
    for y_desired in (0, 1):
        anchors, diag = stageA_target_anchors_from_p(
            p_train=p_train, y_desired=y_desired,
            anchor_k=anchor_k, random_state=anchors_random_state,
        )
        anchors_by_class[y_desired] = anchors
        anchors_diag_by_class[y_desired] = diag

    return GuidedCFCacheAblation(
        X_tr_sc=X_tr_sc,
        predict_proba_guidance=predict_proba_guidance,
        feature_info=feature_info,
        p_train=p_train,
        A=A,
        anchors_by_class=anchors_by_class,
        anchors_diag_by_class=anchors_diag_by_class,
        gamma=gamma,
    )


def pace_with_params_ablation(
    *,
    x_f_sc,
    cache: GuidedCFCacheAblation,
    predict_proba_target,
    n_candidates: int = 2000,
    max_changed_features: int = 8,
    base_sigma: float = 0.25,
    sigma_max: float = 1.0,
    alpha_range=(0.5, 1.0),
    frac_anchor_mix: float = 0.50,
    frac_boundary_mix: float = 0.35,
    frac_guided_noise: float = 0.15,
    order: str = "l0_l2",
    random_state: int = 0,
):
    """Flexible PACE call exposing every A1/A2 generation + selection
    hyperparameter: M, s, sigma_base, alpha_range, pi_a/pi_b/pi_n (A1), and
    order (A2, forwarded to stageC_select_best). gamma lives on the cache
    (Stage A), not here, since it governs unit_weights computed once per
    (dataset, model, seed) -- see guided_cf_build_cache_ablation.
    """
    p_f = float(predict_proba_target(x_f_sc[None, :])[0])
    y_desired = 1 - int(p_f >= 0.5)

    anchors = cache.anchors_by_class[y_desired]
    if anchors.size == 0:
        return None, {"status": "no_anchors", "n_flip": 0, "n_candidates": 0}

    C_sc, meta = stageB_generate(
        x_f=x_f_sc,
        X_train=cache.X_tr_sc,
        predict_proba_fn=cache.predict_proba_guidance,
        boundary_idx=cache.A.boundary_idx,
        anchors_idx=anchors,
        feature_weights=cache.A.feature_weights,
        unit_weights=getattr(cache.A, "unit_weights", None),
        feature_info=cache.feature_info,
        n_candidates=n_candidates,
        max_changed_features=max_changed_features,
        base_sigma=base_sigma,
        sigma_max=sigma_max,
        alpha_range=alpha_range,
        frac_anchor_mix=frac_anchor_mix,
        frac_boundary_mix=frac_boundary_mix,
        frac_guided_noise=frac_guided_noise,
        random_state=random_state,
    )

    x_cf_sc, rep = stageC_select_best(
        x_f=x_f_sc,
        C=C_sc,
        sources=meta["sources"],
        predict_proba_fn=predict_proba_target,
        y_desired=y_desired,
        feature_info=cache.feature_info,
        order=order,
    )
    return x_cf_sc, rep
