"""
MCCE wrapper for the benchmark.

MCCE (Monte Carlo Counterfactual Explanations) fits CART trees to model the
joint distribution of mutable features conditioned on immutables, then samples
candidates and filters for ones that flip the model's prediction.

We work entirely in the preprocessed (scaled + OHE) space so the model wrapper
is trivial and no round-tripping through the preprocessor is needed at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from mcce.mcce import MCCE


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class _MCCEDataset:
    """
    Minimal adapter that satisfies what MCCE.__init__ and metrics.distance need.

    We work in the fully-processed space where every feature is numeric, so
    categorical/immutables lists can be empty (or hold only the immutable cols
    that should be held fixed during generation).
    """

    def __init__(self, feature_names: list[str], immutables_encoded: list[str]):
        self.continuous = list(feature_names)   # all features treated as continuous
        self.categorical = []
        self.categorical_encoded = []
        self.immutables = []
        self.immutables_encoded = list(immutables_encoded)
        self.target = None                      # no outcome column in processed space


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class _ModelWrapper:
    """
    Wraps a sklearn classifier so MCCE can call .predict(DataFrame).

    MCCE filters candidates with  model.predict(cfs) >= 0.5.
    When flip=True the wrapper returns 1-pred so that threshold still selects
    the *minority* class (needed when y_desired == 0).
    """

    def __init__(self, model, flip: bool = False):
        self._model = model
        self._flip = flip

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        proba = self._model.predict_proba(X_df.values)[:, 1]
        pred = (proba >= 0.5).astype(float)
        return 1.0 - pred if self._flip else pred


# ---------------------------------------------------------------------------
# Helper: extract processed feature names from ColumnTransformer
# ---------------------------------------------------------------------------

def _processed_feature_names(pre, num_cols: list[str], cat_cols: list[str]) -> list[str]:
    """
    Return feature names in the order they appear after ColumnTransformer.transform().
    Numeric cols come first (1-1), then OHE-expanded categoricals.
    """
    if pre is None:
        return list(num_cols)

    names: list[str] = []

    if num_cols and "num" in pre.named_transformers_:
        names.extend(num_cols)

    if cat_cols and "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        try:
            names.extend(ohe.get_feature_names_out(cat_cols).tolist())
        except TypeError:
            names.extend(ohe.get_feature_names(cat_cols).tolist())

    return names


# ---------------------------------------------------------------------------
# Output dataclass (mirrors NICEOut / guided output convention)
# ---------------------------------------------------------------------------

@dataclass
class MCCEOut:
    x_cf: Optional[np.ndarray]
    rep: dict
    meta: dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_mcce_explainer(
    X_tr_sc: np.ndarray,
    y_tr: np.ndarray,
    model,
    pre,
    num_cols: list[str],
    cat_cols: list[str],
    immutables=None,
    seed: int = 0,
) -> tuple:
    """
    Fit an MCCE explainer on the preprocessed training data.

    Parameters
    ----------
    X_tr_sc       : (n, d) numpy array  — preprocessed training data
    y_tr          : (n,) numpy array    — training labels (unused by MCCE fit, kept for API symmetry)
    model         : fitted sklearn classifier with predict_proba
    pre           : fitted ColumnTransformer from make_preprocessor
    num_cols      : original continuous feature names
    cat_cols      : original categorical feature names
    immutables    : ImmutableSpec or None
    seed          : random seed

    Returns
    -------
    mcce     : fitted MCCE object
    mcce_ctx : dict with 'feat_names' and 'dtypes' needed by mcce_cf_one
    """
    feat_names = _processed_feature_names(pre, list(num_cols), list(cat_cols))
    n_feats = X_tr_sc.shape[1]
    if len(feat_names) != n_feats:
        feat_names = [f"x{i}" for i in range(n_feats)]

    # Map immutable original names → processed-space names
    imm_num_raw = set(getattr(immutables, "num", set())) & set(num_cols) if immutables else set()
    imm_cat_raw = set(getattr(immutables, "cat", set())) & set(cat_cols) if immutables else set()

    imm_encoded: list[str] = list(imm_num_raw)   # numeric: name preserved 1-1
    for raw_cat in imm_cat_raw:
        for name in feat_names:
            if name.startswith(f"{raw_cat}_") or name == raw_cat:
                imm_encoded.append(name)

    dataset = _MCCEDataset(feat_names, imm_encoded)
    X_tr_df = pd.DataFrame(X_tr_sc, columns=feat_names)
    dtypes = {col: "float64" for col in feat_names}

    mcce = MCCE(dataset=dataset, model=_ModelWrapper(model), seed=seed)
    mcce.fit(X_tr_df, dtypes)

    return mcce, {"feat_names": feat_names, "dtypes": dtypes}


def mcce_cf_one(
    *,
    mcce_obj: MCCE,
    mcce_ctx: dict,
    x_f_sc: np.ndarray,
    y_desired: int,
    model,
    proba_threshold: float = 0.5,
    k: int = 1000,
) -> MCCEOut:
    """
    Generate one counterfactual for x_f_sc using a fitted MCCE object.

    Parameters
    ----------
    mcce_obj        : fitted MCCE (shared across factuals for the same model)
    mcce_ctx        : dict returned by build_mcce_explainer
    x_f_sc          : 1-D numpy array in preprocessed (model) space
    y_desired       : desired predicted class (0 or 1)
    model           : fitted sklearn classifier
    proba_threshold : decision threshold
    k               : candidates sampled per leaf node
    """
    feat_names = mcce_ctx["feat_names"]
    x_f_df = pd.DataFrame(x_f_sc.reshape(1, -1), columns=feat_names)

    # Swap model wrapper direction: flip=True makes >=0.5 select class-0 CFs
    mcce_obj.model = _ModelWrapper(model, flip=(y_desired == 0))

    try:
        cfs = mcce_obj.generate(x_f_df, k=k)
        mcce_obj.postprocess(cfs, x_f_df, cutoff=0.5)

        rs = getattr(mcce_obj, "results_sparse", None)
        if rs is None or len(rs) == 0:
            return MCCEOut(
                x_cf=None,
                rep={"ok": False, "status": "no_cf"},
                meta={"method": "mcce"},
            )

        cf_row = rs.iloc[0]
        x_cf = cf_row[feat_names].values.astype(np.float64)

        p_cf = float(model.predict_proba(x_cf.reshape(1, -1))[0, 1])
        if int(p_cf >= proba_threshold) != y_desired:
            return MCCEOut(
                x_cf=None,
                rep={"ok": False, "status": "flip_failed", "p_cf": p_cf},
                meta={"method": "mcce"},
            )

        return MCCEOut(
            x_cf=x_cf,
            rep={"ok": True, "status": "ok"},
            meta={
                "method": "mcce",
                "nb_unique_pos": int(cf_row.get("nb_unique_pos", -1)),
            },
        )

    except Exception as e:
        return MCCEOut(
            x_cf=None,
            rep={"ok": False, "status": "error", "msg": str(e)},
            meta={"method": "mcce"},
        )
