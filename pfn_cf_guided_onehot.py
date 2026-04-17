# pfn_cf_guided_onehot.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np

PredictProbaFn = Callable[[np.ndarray], np.ndarray]  # returns (n,) probabilities for class 1

@dataclass
class StageAArtifacts:
    boundary_idx: np.ndarray
    feature_weights: np.ndarray          # (d,) legacy per-column weights
    unit_weights: Optional[np.ndarray]   # (n_units,) weights for [num cols + cat groups]
    diag: Dict[str, Any]


@dataclass(frozen=True)
class FeatureInfo:
    """
    Model-space feature metadata.

    - num_idx: indices of continuous features (optional; if None we infer as "all non-cat columns")
    - cat_groups: list of arrays; each array contains the one-hot column indices for one categorical variable
    """
    cat_groups: List[np.ndarray]
    num_idx: Optional[np.ndarray] = None

    immutable_num: frozenset[int] = frozenset()
    immutable_cat: frozenset[int] = frozenset() 

def _perm_importance_proba_drop(
    predict_proba_fn: PredictProbaFn,
    X_ref: np.ndarray,
    *,
    n_repeats: int,
    rng: np.random.Generator,
    feature_info: Optional[FeatureInfo] = None,
    aggregate: str = "none",
) -> np.ndarray:
    """
    Permutation importance on p(x): mean abs change in predicted probabilities after permuting.

    If feature_info with cat_groups is provided:
      - permutes each categorical group as a block to preserve one-hot validity.
      - supports per-unit importances ("unit") or per-column importances ("none").

    Returns:
      - aggregate="none": array of shape (d,) column-level importances (cat group importance spread across its columns)
      - aggregate="unit": array of shape (n_units,) where units are [num cols..., cat groups...]
    """
    X_ref = np.asarray(X_ref)
    if X_ref.ndim != 2:
        raise ValueError(f"X_ref must be 2D, got shape {X_ref.shape}")
    if n_repeats <= 0:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if aggregate not in ("none", "unit"):
        raise ValueError(f"aggregate must be 'none' or 'unit', got {aggregate!r}")

    # Baseline once
    p0 = predict_proba_fn(X_ref).astype(np.float64)
    n, d = X_ref.shape

    cat_groups: List[np.ndarray] = []
    num_idx: Optional[np.ndarray] = None
    if feature_info is not None:
        cat_groups = [np.asarray(g, dtype=int) for g in feature_info.cat_groups]
        num_idx = feature_info.num_idx

    # Infer numeric indices as "all columns not in any cat group" if not provided
    if num_idx is None:
        cat_cols = np.zeros(d, dtype=bool)
        for g in cat_groups:
            if g.size:
                cat_cols[g] = True
        num_idx = np.flatnonzero(~cat_cols)
    else:
        num_idx = np.asarray(num_idx, dtype=int)

    # Build permutation "units" (same as you)
    units: List[np.ndarray] = []
    for j in num_idx:
        units.append(np.array([int(j)], dtype=int))
    for g in cat_groups:
        units.append(g.copy())

    unit_imps = np.zeros(len(units), dtype=np.float64)

    # --- batched evaluation per unit ---
    # For each unit, build (n_repeats * n, d) matrix and evaluate predict_proba once.
    for u, cols in enumerate(units):
        # Make n_repeats copies of X_ref into one big batch
        X_stack = np.repeat(X_ref, repeats=n_repeats, axis=0).copy()

        # Apply permutations per repeat to the slice [r*n : (r+1)*n]
        for r in range(n_repeats):
            sl = slice(r * n, (r + 1) * n)
            perm = rng.permutation(n)
            X_stack[sl, cols] = X_stack[sl, :][perm][:, cols]

        # Single model call for all repeats of this unit
        p_stack = predict_proba_fn(X_stack).astype(np.float64)

        # Reshape to (n_repeats, n, ...) and compare to p0 (broadcast)
        # Works for binary (n,2), multiclass (n,K), or even (n,) if your fn returns 1D.
        p0_b = p0[None, ...]
        try:
            p_stack = p_stack.reshape((n_repeats,) + p0.shape)
        except ValueError as e:
            raise ValueError(
                f"predict_proba_fn returned shape {p_stack.shape} for stacked input "
                f"but baseline shape is {p0.shape}; cannot reshape to "
                f"{(n_repeats,) + p0.shape}. Your predict_proba_fn must be consistent."
            ) from e

        delta = np.abs(p_stack - p0_b)
        unit_imps[u] = float(delta.mean())

    if aggregate == "unit":
        return unit_imps

    # aggregate == "none" -> return per-column importances
    imps = np.zeros(d, dtype=np.float64)

    # numeric cols: map unit importance back to that column
    u = 0
    for j in num_idx:
        imps[int(j)] = unit_imps[u]
        u += 1

    # cat groups: spread importance across columns uniformly
    for g in cat_groups:
        g_imp = unit_imps[u]
        u += 1
        if len(g) > 0:
            imps[g] = g_imp / float(len(g))

    return imps

# def _perm_importance_proba_drop(
#     predict_proba_fn: PredictProbaFn,
#     X_ref: np.ndarray,
#     *,
#     n_repeats: int,
#     rng: np.random.Generator,
#     feature_info: Optional[FeatureInfo] = None,
#     aggregate: str = "none", 
# ) -> np.ndarray:
#     """
#     Permutation importance on p(x): mean abs change in predicted probabilities after permuting.

#     If feature_info with cat_groups is provided:
#       - permutes each categorical group as a block to preserve one-hot validity.
#       - supports per-unit importances ("unit") or per-column importances ("none").

#     Returns:
#       - aggregate="none": array of shape (d,) column-level importances (cat group importance spread across its columns)
#       - aggregate="unit": array of shape (n_units,) where units are [num cols..., cat groups...]
#     """
#     p0 = predict_proba_fn(X_ref).astype(np.float64)
#     n, d = X_ref.shape

#     cat_groups: List[np.ndarray] = []
#     num_idx: Optional[np.ndarray] = None
#     if feature_info is not None:
#         cat_groups = [np.asarray(g, dtype=int) for g in feature_info.cat_groups]
#         num_idx = feature_info.num_idx

#     # Infer numeric indices as "all columns not in any cat group" if not provided
#     if num_idx is None:
#         cat_cols = np.zeros(d, dtype=bool)
#         for g in cat_groups:
#             cat_cols[g] = True
#         num_idx = np.flatnonzero(~cat_cols)

#     # Build permutation "units"
#     # Each unit is either a single numeric column, or a categorical group (array of columns)
#     units: List[Tuple[str, np.ndarray]] = []
#     for j in num_idx:
#         units.append(("num", np.array([int(j)], dtype=int)))
#     for g in cat_groups:
#         units.append(("cat", g.copy()))

#     unit_imps = np.zeros(len(units), dtype=np.float64)

#     for u, (_kind, cols) in enumerate(units):
#         deltas = np.empty(n_repeats, dtype=np.float64)
#         for r in range(n_repeats):
#             Xp = X_ref.copy()
#             perm = rng.permutation(n)
#             # permute whole block for categoricals, single col for numerics
#             Xp[:, cols] = Xp[perm][:, cols]
#             pj = predict_proba_fn(Xp).astype(np.float64)
#             deltas[r] = np.mean(np.abs(pj - p0))
#         unit_imps[u] = float(deltas.mean())

#     if aggregate == "unit":
#         return unit_imps

#     # aggregate == "none" -> return per-column importances (backward compatible)
#     imps = np.zeros(d, dtype=np.float64)

#     # numeric cols: map unit importance back to that column
#     u = 0
#     for j in num_idx:
#         imps[int(j)] = unit_imps[u]
#         u += 1

#     # cat groups: spread the group's importance across its columns (uniformly)
#     for g in cat_groups:
#         g_imp = unit_imps[u]
#         u += 1
#         if len(g) > 0:
#             imps[g] = g_imp / float(len(g))

#     return imps


def stageA_build(
    *,
    X_train: np.ndarray,
    predict_proba_fn: PredictProbaFn,
    boundary_k: int = 200,
    boundary_balance: bool = True,
    perm_repeats: int = 8,
    gamma: float = 1.0,
    random_state: int = 0,
    feature_info: Optional[FeatureInfo] = None,   # NEW
) -> StageAArtifacts:
    """
    Precompute:
      - boundary pool indices (p ~ 0.5)
      - feature sampling weights from permutation importance near boundary

    With feature_info.cat_groups:
      - permutation importance is computed per-unit (num cols + cat groups),
        preserving one-hot validity.
      - returns both unit_weights (recommended) and feature_weights (legacy per-column).
    """
    rng = np.random.default_rng(random_state)

    p = predict_proba_fn(X_train).astype(np.float64)
    margin = np.abs(p - 0.5)
    order = np.argsort(margin)

    if not boundary_balance:
        boundary_idx = order[: min(boundary_k, len(order))]
    else:
        yhat = (p >= 0.5).astype(int)
        idx0 = order[yhat[order] == 0]
        idx1 = order[yhat[order] == 1]
        k0 = min(boundary_k // 2, len(idx0))
        k1 = min(boundary_k - k0, len(idx1))
        boundary_idx = np.concatenate([idx0[:k0], idx1[:k1]])
        if boundary_idx.size == 0:
            boundary_idx = order[: min(boundary_k, len(order))]
        rng.shuffle(boundary_idx)

    X_ref = X_train[boundary_idx]

    # --- NEW: compute importance per-unit if categoricals exist ---
    if feature_info is not None and len(feature_info.cat_groups) > 0:
        raw_unit = _perm_importance_proba_drop(
            predict_proba_fn,
            X_ref,
            n_repeats=perm_repeats,
            rng=rng,
            feature_info=feature_info,
            aggregate="unit",
        ).astype(np.float64)

        raw_unit = np.maximum(raw_unit, 1e-12) ** gamma
        unit_weights = raw_unit / raw_unit.sum()

        # Also keep legacy per-column weights for anything downstream still expecting (d,)
        raw_col = _perm_importance_proba_drop(
            predict_proba_fn,
            X_ref,
            n_repeats=perm_repeats,
            rng=rng,
            feature_info=feature_info,
            aggregate="none",
        ).astype(np.float64)

        raw_col = np.maximum(raw_col, 1e-12) ** gamma
        feature_weights = raw_col / raw_col.sum()

        diag_raw = {
            "raw_importance_unit": raw_unit,
            "unit_weights": unit_weights,
            "raw_importance_col": raw_col,
            "feature_weights_col": feature_weights,
            "n_units": int(len(unit_weights)),
            "n_cat_groups": int(len(feature_info.cat_groups)),
        }

    else:
        # --- Original behavior: per-column only ---
        raw = _perm_importance_proba_drop(
            predict_proba_fn, X_ref, n_repeats=perm_repeats, rng=rng
        ).astype(np.float64)

        raw = np.maximum(raw, 1e-12) ** gamma
        feature_weights = raw / raw.sum()
        unit_weights = None

        diag_raw = {
            "raw_importance": raw,
            "feature_weights": feature_weights,
            "n_units": int(X_train.shape[1]),
            "n_cat_groups": 0,
        }

    diag = {
        "boundary_k_actual": int(len(boundary_idx)),
        "p_boundary_min": float(p[boundary_idx].min()) if len(boundary_idx) else np.nan,
        "p_boundary_max": float(p[boundary_idx].max()) if len(boundary_idx) else np.nan,
        "margin_median": float(np.median(margin[boundary_idx])) if len(boundary_idx) else np.nan,
        **diag_raw,
    }

    return StageAArtifacts(
        boundary_idx=np.asarray(boundary_idx, dtype=int),
        feature_weights=np.asarray(feature_weights, dtype=np.float64),
        unit_weights=None if unit_weights is None else np.asarray(unit_weights, dtype=np.float64),
        diag=diag,
    )


def stageA_target_anchors_from_p(
    *,
    p_train: np.ndarray,
    y_desired: int,
    anchor_k: int = 200,
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pick near-boundary anchors on the desired side (using cached probs)."""
    rng = np.random.default_rng(random_state)

    p = p_train.astype(np.float64)
    yhat = (p >= 0.5).astype(int)

    idx = np.where(yhat == y_desired)[0]
    if idx.size == 0:
        return np.array([], dtype=int), {"anchor_k_actual": 0}

    margin = np.abs(p[idx] - 0.5)
    order = idx[np.argsort(margin)]
    anchors = order[: min(anchor_k, len(order))]
    rng.shuffle(anchors)

    diag = {
        "anchor_k_actual": int(len(anchors)),
        "p_anchor_min": float(p[anchors].min()),
        "p_anchor_max": float(p[anchors].max()),
        "margin_anchor_median": float(np.median(np.abs(p[anchors] - 0.5))),
    }
    return anchors.astype(int), diag


def stageB_generate(
    *,
    x_f: np.ndarray,
    predict_proba_fn: PredictProbaFn,
    X_train: np.ndarray,
    boundary_idx: np.ndarray,
    anchors_idx: np.ndarray,
    feature_weights: np.ndarray,                 # legacy (d,)
    unit_weights: Optional[np.ndarray] = None,   # NEW (n_units,), preferred
    feature_info: Optional[FeatureInfo] = None,  # NEW
    n_candidates: int = 800,
    max_changed_features: int = 3,
    p_change: float = 0.8,
    alpha_range: Tuple[float, float] = (0.5, 1.0), #(0.1, 0.8),
    base_sigma: float = 0.25,
    sigma_min: float = 0.05,
    sigma_max: float = 1.0,
    frac_anchor_mix: float = 0.50,
    frac_boundary_mix: float = 0.35,
    frac_guided_noise: float = 0.15,
    clip_to_train_quantiles: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
    # categorical noise behaviour:
    cat_flip_prob: float = 1.0,                  # in guided_noise, flip selected cat group with this prob
    random_state: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a candidate set around x_f using guidance from Stage A (supports one-hot categoricals)."""
    rng = np.random.default_rng(random_state)
    x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
    d = x_f.shape[0]

    # --- Build numeric indices + cat groups (if provided) ---
    cat_groups: List[np.ndarray] = []
    num_idx: np.ndarray

    if feature_info is not None and len(feature_info.cat_groups) > 0:
        cat_groups = [np.asarray(g, dtype=int) for g in feature_info.cat_groups]

        if feature_info.num_idx is not None:
            num_idx = np.asarray(feature_info.num_idx, dtype=int)
        else:
            cat_cols = np.zeros(d, dtype=bool)
            for g in cat_groups:
                cat_cols[g] = True
            num_idx = np.flatnonzero(~cat_cols)
    else:
        num_idx = np.arange(d, dtype=int)

    # Probability at factual + sigma schedule
    p_f = float(predict_proba_fn(x_f[None, :])[0])
    margin = abs(p_f - 0.5)
    scale = np.clip(margin / 0.25, 0.2, 2.5)
    sigma = float(np.clip(base_sigma * scale, sigma_min, sigma_max))

    # --- Quantile clipping for numeric columns only ---
    if clip_to_train_quantiles and num_idx.size > 0:
        lo_full = np.quantile(X_train, q_low, axis=0)
        hi_full = np.quantile(X_train, q_high, axis=0)
        lo_num = lo_full[num_idx]
        hi_num = hi_full[num_idx]
    else:
        lo_num = hi_num = None

    def clip_numeric_(x: np.ndarray) -> np.ndarray:
        if lo_num is None:
            return x
        xn = x[num_idx]
        xn = np.minimum(np.maximum(xn, lo_num), hi_num)
        x[num_idx] = xn
        return x

    # --- Build "units": numerics are singletons; cats are groups ---
    # immutability sets
    immutable_num = feature_info.immutable_num if (feature_info is not None) else frozenset()
    immutable_cat = feature_info.immutable_cat if (feature_info is not None) else frozenset()

    # Full units in the SAME order as Stage A uses:
    full_units = []
    for j in num_idx:
        full_units.append(("num", int(j)))
    for g_i in range(len(cat_groups)):
        full_units.append(("cat", int(g_i)))

    # Actionable units (the ones Stage B will actually use), and their indices in full_units
    actionable_full_indices = []
    units = []

    # numeric units
    for u, (kind, key) in enumerate(full_units[:len(num_idx)]):
        j = key
        if j in immutable_num:
            continue
        actionable_full_indices.append(u)
        units.append(("num", np.array([j], dtype=int)))

    # categorical units
    for g_i, g in enumerate(cat_groups):
        u = len(num_idx) + g_i
        if g_i in immutable_cat:
            continue
        actionable_full_indices.append(u)
        units.append(("cat", (g_i, g.copy())))

    n_units = len(units)

    # --- Unit sampling weights ---
    if unit_weights is not None:
        uw_full = np.asarray(unit_weights, dtype=np.float64).reshape(-1)
        if uw_full.size != len(full_units):
            raise ValueError(
                f"unit_weights has shape {uw_full.shape}, expected ({len(full_units)},) "
                f"(full units = num({len(num_idx)}) + cat({len(cat_groups)}))"
            )
        uw = uw_full[np.asarray(actionable_full_indices, dtype=int)]
        uw = np.maximum(uw, 1e-12)
        uw = uw / uw.sum()
    else:
        # fallback: derive unit weights from per-column feature_weights
        fw = np.asarray(feature_weights, dtype=np.float64).reshape(-1)
        if fw.size != d:
            raise ValueError(f"feature_weights has shape {fw.shape}, expected ({d},)")
        fw = np.maximum(fw, 1e-12)
        fw = fw / fw.sum()

        uw_list = []
        # numeric unit weight = fw[col] — only actionable (non-immutable) units
        for j in num_idx:
            if int(j) in immutable_num:
                continue
            uw_list.append(float(fw[int(j)]))
        # cat group unit weight = sum of fw over its columns — only actionable groups
        for g_i, g in enumerate(cat_groups):
            if g_i in immutable_cat:
                continue
            uw_list.append(float(fw[g].sum()))
        uw = np.asarray(uw_list, dtype=np.float64)
        uw = np.maximum(uw, 1e-12)
        uw = uw / uw.sum()

    # --- Categorical category probabilities (optional, but helps guided_noise) ---
    # If not available, use uniform for each group.
    cat_probs: List[np.ndarray] = []
    if cat_groups:
        for g in cat_groups:
            # empirical probs from training one-hot means
            pr = np.mean(X_train[:, g], axis=0).astype(np.float64)
            pr = np.maximum(pr, 1e-12)
            pr = pr / pr.sum()
            cat_probs.append(pr)

    def sample_units() -> List[int]:
        """Sample up to max_changed_features units, returning their indices in `units`."""
        k = int(rng.integers(1, max_changed_features + 1))
        if rng.random() > p_change:
            k = 0
        if k == 0:
            return []
        # choose distinct units
        return rng.choice(n_units, size=min(k, n_units), replace=False, p=uw).tolist()

    def apply_anchor_or_boundary_mix(x: np.ndarray, donor: np.ndarray, chosen_units: List[int]) -> np.ndarray:
        """Apply mix operator: numeric mix+noise; categorical copy donor one-hot."""
        if not chosen_units:
            return x
        alpha = float(rng.uniform(*alpha_range))

        for ui in chosen_units:
            kind, payload = units[ui]
            if kind == "num":
                cols = payload
                j = int(cols[0])
                x[j] = (1 - alpha) * x_f[j] + alpha * donor[j]
                x[j] += float(rng.normal(0.0, sigma))
            else:
                g_i, cols = payload
                x[cols] = donor[cols]
        return x

    def apply_guided_noise(x: np.ndarray, chosen_units: List[int]) -> np.ndarray:
        """Apply noise operator: numeric gaussian; categorical flip to another category."""
        if not chosen_units:
            return x

        for ui in chosen_units:
            kind, payload = units[ui]
            if kind == "num":
                cols = payload
                j = int(cols[0])
                x[j] += float(rng.normal(0.0, sigma))
            else:
                if rng.random() > cat_flip_prob:
                    continue
                g_i, cols = payload
                probs = cat_probs[g_i]
                # defensive guard (keep during debugging)
                if probs.shape[0] != len(cols):
                    raise ValueError(f"cat_probs mismatch for group {g_i}: probs={probs.shape[0]} cols={len(cols)}")
                new_pos = int(rng.choice(len(cols), p=probs))
                x[cols] = 0.0
                x[int(cols[new_pos])] = 1.0
        return x

    # --- Mix proportions ---
    fracs = np.array([frac_anchor_mix, frac_boundary_mix, frac_guided_noise], dtype=np.float64)
    fracs = np.maximum(fracs, 0.0)
    if fracs.sum() == 0:
        fracs = np.array([1.0, 0.0, 0.0])
    fracs /= fracs.sum()

    n_anchor = int(round(n_candidates * fracs[0]))
    n_boundary = int(round(n_candidates * fracs[1]))
    n_noise = n_candidates - n_anchor - n_boundary

    C_list = []
    sources = []

    # anchor_mix
    if anchors_idx is None or len(anchors_idx) == 0:
        n_boundary += n_anchor
        n_anchor = 0

    for _ in range(n_anchor):
        j = int(rng.choice(anchors_idx))
        donor = X_train[j].astype(np.float64, copy=False)
        chosen = sample_units()
        x = x_f.copy()
        x = apply_anchor_or_boundary_mix(x, donor, chosen)
        x = clip_numeric_(x)
        C_list.append(x)
        sources.append("anchor_mix")

    # boundary_mix
    for _ in range(n_boundary):
        j = int(rng.choice(boundary_idx))
        donor = X_train[j].astype(np.float64, copy=False)
        chosen = sample_units()
        x = x_f.copy()
        x = apply_anchor_or_boundary_mix(x, donor, chosen)
        x = clip_numeric_(x)
        C_list.append(x)
        sources.append("boundary_mix")

    # guided_noise
    for _ in range(n_noise):
        chosen = sample_units()
        x = x_f.copy()
        x = apply_guided_noise(x, chosen)
        x = clip_numeric_(x)
        C_list.append(x)
        sources.append("guided_noise")

    C = np.vstack(C_list).astype(np.float32)

    meta = {
        "sources": np.array(sources, dtype=object),
        "info": {
            "p_f": p_f,
            "margin": margin,
            "sigma": sigma,
            "n_units": int(n_units),
            "n_num_units_actionable": int(sum(1 for k, _ in units if k == "num")),
            "n_cat_units_actionable": int(sum(1 for k, _ in units if k == "cat")),
            "n_cat_groups_total": int(len(cat_groups)),
        },
    }
    return C, meta


# -------------------------
# Stage C (selection)
# -------------------------
def onehot_valid_mask(C: np.ndarray, cat_groups: list[np.ndarray], tol: float = 1e-6) -> np.ndarray:
    ok = np.ones(C.shape[0], dtype=bool)
    for g in cat_groups:
        s = C[:, g].sum(axis=1)
        ok &= np.isclose(s, 1.0, atol=tol)
        ok &= (C[:, g] >= -tol).all(axis=1)
        ok &= (C[:, g] <= 1.0 + tol).all(axis=1)
    return ok


def l0_group_aware(
    x_f: np.ndarray,
    C: np.ndarray,
    *,
    num_idx: np.ndarray,
    cat_groups: list[np.ndarray],
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Count changed *units*:
      - numeric: per-column change
      - categorical: per-group change (any column differs -> counts as 1)
    """
    x_f = np.asarray(x_f)
    C = np.asarray(C)

    # numeric changes
    if num_idx.size > 0:
        num_changed = (np.abs(C[:, num_idx] - x_f[None, num_idx]) > tol).sum(axis=1)
    else:
        num_changed = np.zeros(C.shape[0], dtype=int)

    # categorical group changes
    cat_changed = np.zeros(C.shape[0], dtype=int)
    for g in cat_groups:
        # any difference in the group -> 1
        diff = np.any(np.abs(C[:, g] - x_f[None, g]) > tol, axis=1)
        cat_changed += diff.astype(int)

    return num_changed + cat_changed

def stageC_select_best(
    *,
    x_f: np.ndarray,
    C: np.ndarray,
    predict_proba_fn: PredictProbaFn,
    y_desired: int,
    l0_tol: float = 1e-6,
    proba_threshold: float = 0.5,
    require_margin: Optional[float] = None,
    feature_info: Optional[FeatureInfo] = None,
    sources: Optional[np.ndarray] = None, 
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:

    x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
    C = np.asarray(C, dtype=np.float64)

    n0 = int(len(C))
    if sources is not None:
        sources = np.asarray(sources)
        if sources.shape[0] != n0:
            raise ValueError(f"sources length {sources.shape[0]} != len(C) {n0}")

    if n0 == 0:
        return None, {"status": "no_candidates", "n_candidates": 0, "n_flip": 0}

    # validity filtering
    if feature_info is not None and len(getattr(feature_info, "cat_groups", [])) > 0:
        cat_groups = [np.asarray(g, dtype=int) for g in feature_info.cat_groups]
        ok = onehot_valid_mask(C, cat_groups, tol=l0_tol)
        if not np.all(ok):
            C = C[ok]
            if sources is not None:
                sources = sources[ok]
        if len(C) == 0:
            return None, {"status": "no_valid_candidates", "n_candidates": n0, "n_valid": 0, "n_flip": 0}
    else:
        cat_groups = []

    # --- compute proba and flip on filtered C ---
    # --- optional: deduplicate candidates to avoid wasted predict_proba work ---
    # Works best because StageB returns float32; rounding makes equality stable.
    dedup_decimals = 6  # tweak: 5–7 typical
    if C.shape[0] > 1:
        C_key = np.round(C, dedup_decimals)
        C_unique, inv = np.unique(C_key, axis=0, return_inverse=True)

        # If we reduced size, keep the corresponding true-valued rows
        if C_unique.shape[0] < C.shape[0]:
            # map unique rows back to one representative from original C
            # (use first occurrence)
            first = np.zeros(C_unique.shape[0], dtype=int)
            first.fill(-1)
            for i, u in enumerate(inv):
                if first[u] == -1:
                    first[u] = i
            C_u = C[first]

            # Predict only on uniques
            p_u = predict_proba_fn(C_u).astype(np.float64)

            # Expand back to original length if you want 1:1 candidate accounting
            pC = p_u[inv]
        else:
            pC = predict_proba_fn(C).astype(np.float64)
    else:
        pC = predict_proba_fn(C).astype(np.float64)

    yhatC = (pC >= proba_threshold).astype(int)
    flip = (yhatC == y_desired)

    n_flip = int(np.sum(flip))
    if n_flip == 0:
        return None, {"status": "no_flip", "n_candidates": int(len(C)), "n_flip": 0}

    Cf = C[flip]
    pf = pC[flip]
    if sources is not None:
        sf = sources[flip]

    # --- group-aware L0 and (simple) L2 ---
    if feature_info is not None and len(cat_groups) > 0:
        d = x_f.shape[0]
        if feature_info.num_idx is not None:
            num_idx = np.asarray(feature_info.num_idx, dtype=int)
        else:
            cat_cols = np.zeros(d, dtype=bool)
            for g in cat_groups:
                cat_cols[g] = True
            num_idx = np.flatnonzero(~cat_cols)

        l0 = l0_group_aware(x_f, Cf, num_idx=num_idx, cat_groups=cat_groups, tol=l0_tol).astype(int)
    else:
        l0 = np.sum(np.abs(Cf - x_f[None, :]) > l0_tol, axis=1).astype(int)

    l2 = np.sqrt(np.sum((Cf - x_f[None, :]) ** 2, axis=1)).astype(np.float64)

    # --- margin constraint, if requested ---
    if require_margin is not None:
        if y_desired == 1:
            strong = pf >= require_margin
        else:
            strong = pf <= require_margin
        if np.any(strong):
            Cf, pf, l0, l2 = Cf[strong], pf[strong], l0[strong], l2[strong]
            if sources is not None:
                sf = sf[strong]
            #Cf, pf, l0, l2 = Cf[strong], pf[strong], l0[strong], l2[strong]

    order = np.lexsort((np.abs(pf - 0.5), l2, l0))
    best = int(order[0])
    x_cf = Cf[best].astype(np.float32, copy=False)

    rep = {
        "status": "ok",
        "n_candidates": int(len(C)),
        "n_flip": int(np.sum(flip)),
        "l0": int(l0[best]),
        "l2": float(l2[best]),
        "p_cf": float(pf[best]),
    }
    if sources is not None:
        rep["best_source"] = str(sf[best])

    return x_cf, rep


# def stageC_select_best(
#     *,
#     x_f: np.ndarray,
#     C: np.ndarray,
#     predict_proba_fn: PredictProbaFn,
#     y_desired: int,
#     l0_tol: float = 1e-6,
#     proba_threshold: float = 0.5,
#     require_margin: Optional[float] = None,
#     feature_info: Optional[FeatureInfo] = None, 
# ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
#     """
#     Select best CF:
#       - must flip
#       - lexicographic: min L0 then min L2 then min |p-0.5|
#       - optional: enforce a proba margin for "strong" CFs
#       - (optional) validate one-hot categoricals
#     """
#     x_f = np.asarray(x_f, dtype=np.float64).reshape(-1)
#     C = np.asarray(C, dtype=np.float64)

#     n0 = int(len(C))
#     if n0 == 0:
#         return None, {"status": "no_candidates", "n_candidates": 0, "n_flip": 0}

#     # --- optional validity filtering for one-hot categoricals ---
#     if feature_info is not None and len(getattr(feature_info, "cat_groups", [])) > 0:
#         cat_groups = [np.asarray(g, dtype=int) for g in feature_info.cat_groups]
#         ok = onehot_valid_mask(C, cat_groups, tol=l0_tol)
#         if not np.all(ok):
#             C = C[ok]
#         if len(C) == 0:
#             return None, {
#                 "status": "no_valid_candidates",
#                 "n_candidates": n0,
#                 "n_valid": 0,
#                 "n_flip": 0,
#             }
#     else:
#         cat_groups = []

#     # --- compute proba and flip on filtered C ---
#     # --- optional: deduplicate candidates to avoid wasted predict_proba work ---
#     # Works best because StageB returns float32; rounding makes equality stable.
#     dedup_decimals = 6  # tweak: 5–7 typical
#     if C.shape[0] > 1:
#         C_key = np.round(C, dedup_decimals)
#         C_unique, inv = np.unique(C_key, axis=0, return_inverse=True)

#         # If we reduced size, keep the corresponding true-valued rows
#         if C_unique.shape[0] < C.shape[0]:
#             # map unique rows back to one representative from original C
#             # (use first occurrence)
#             first = np.zeros(C_unique.shape[0], dtype=int)
#             first.fill(-1)
#             for i, u in enumerate(inv):
#                 if first[u] == -1:
#                     first[u] = i
#             C_u = C[first]

#             # Predict only on uniques
#             p_u = predict_proba_fn(C_u).astype(np.float64)

#             # Expand back to original length if you want 1:1 candidate accounting
#             pC = p_u[inv]
#         else:
#             pC = predict_proba_fn(C).astype(np.float64)
#     else:
#         pC = predict_proba_fn(C).astype(np.float64)

#     #pC = predict_proba_fn(C).astype(np.float64)
#     yhatC = (pC >= proba_threshold).astype(int)
#     flip = (yhatC == y_desired)

#     n_flip = int(np.sum(flip))
#     if n_flip == 0:
#         return None, {
#             "status": "no_flip",
#             "n_candidates": int(len(C)),
#             "n_flip": 0,
#         }

#     Cf = C[flip]
#     pf = pC[flip]

#     # --- group-aware L0 and (simple) L2 ---
#     if feature_info is not None and len(cat_groups) > 0:
#         d = x_f.shape[0]
#         if feature_info.num_idx is not None:
#             num_idx = np.asarray(feature_info.num_idx, dtype=int)
#         else:
#             cat_cols = np.zeros(d, dtype=bool)
#             for g in cat_groups:
#                 cat_cols[g] = True
#             num_idx = np.flatnonzero(~cat_cols)

#         l0 = l0_group_aware(x_f, Cf, num_idx=num_idx, cat_groups=cat_groups, tol=l0_tol).astype(int)
#     else:
#         l0 = np.sum(np.abs(Cf - x_f[None, :]) > l0_tol, axis=1).astype(int)

#     l2 = np.sqrt(np.sum((Cf - x_f[None, :]) ** 2, axis=1)).astype(np.float64)

#     # --- margin constraint, if requested ---
#     if require_margin is not None:
#         if y_desired == 1:
#             strong = pf >= require_margin
#         else:
#             strong = pf <= require_margin
#         if np.any(strong):
#             Cf, pf, l0, l2 = Cf[strong], pf[strong], l0[strong], l2[strong]

#     # --- lexicographic selection ---
#     order = np.lexsort((np.abs(pf - 0.5), l2, l0))
#     best = int(order[0])
#     x_cf = Cf[best].astype(np.float32, copy=False)

#     rep = {
#         "status": "ok",
#         "n_candidates": int(len(C)),
#         "n_flip": int(np.sum(flip)),
#         "l0": int(l0[best]),
#         "l2": float(l2[best]),
#         "p_cf": float(pf[best]),
#     }
#     return x_cf, rep