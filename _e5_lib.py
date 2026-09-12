# ==========================================================================
# E5b feasibility test: real DiCE-vs-TabPFN counterfactual generation.
#
# Purpose: check whether a genuinely-built (not stale-RF) DiCE explainer can
# generate counterfactuals against TabPFN in reasonable wall-clock time, on
# a small real-world dataset (blood_transfusion, 748 rows x 4 features),
# using DiCE's "random" method (the one actually used for all real-world
# datasets per choose_dice_method).
#
# This script is extracted verbatim from tabpfn_cf5.ipynb cells
# [1,2,3,5,7,8,10,11,12,13,14,15,16,17,18], with exactly two patches applied
# to fix the E5b stale-dice_exp bug:
#   1. fit_models_for_dice now also builds a TabPFN pipeline
#      (was: only LR/XGB/RF).
#   2. run_benchmark now always rebuilds dice_model/dice_exp per model_name
#      (was: skipped for TabPFN, silently reusing the previous model's
#      explainer -- the confirmed bug), plus an assert guard against the
#      failure mode recurring.
#
# Does NOT touch output_realdata.joblib / output_mockdata.joblib or the
# master notebook. Prints per-instance results and a timing summary.
# ==========================================================================

# ---- cell 1 ----
from __future__ import annotations

import os

# ---- cell 2 ----
import tqdm
import tqdm.auto

# Replace with a no-op
class _NoOpTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, *args, **kwargs): pass
    def close(self): pass

tqdm.tqdm = _NoOpTqdm
tqdm.auto.tqdm = _NoOpTqdm

# ---- cell 3 ----
import numpy as np
import pandas as pd
import time
import zlib
import joblib

from typing import Callable, Optional, Dict, Any, Tuple
from types import SimpleNamespace
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(1, '/nvme/h/lchristodoulou/pace/cf-proposal/')
from pfn_cf_guided_onehot import *
from make_mock_cont import *
from get_real_datasets import *
from preprocessing import *
from mcce_cf import build_mcce_explainer, mcce_cf_one, MCCEOut

from dataclasses import dataclass

from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from nice import NICE
import logging
import dice_ml
from dice_ml import Model
from dice_ml import Dice
logging.getLogger("dice_ml").setLevel(logging.WARNING)
logging.getLogger("analytics").setLevel(logging.CRITICAL)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- cell 5 ----
try:
    from dice_ml.utils.exception import UserConfigValidationException
except ImportError:
    UserConfigValidationException = Exception  # fallback

# ---- cell 7 ----
# -------------------------
# Model wrappers (predict_proba for class 1)
# -------------------------
def fit_models(X_tr_sc, y_tr, seed=0, include_tabpfn=True):
    models = {}

    # LR
    lr = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=seed)
    lr.fit(X_tr_sc, y_tr)
    models["LR"] = lr

    # XGB
    xgb = XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    xgb.fit(X_tr_sc, y_tr)
    models["XGB"] = xgb

    # RF
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    rf.fit(X_tr_sc, y_tr)
    models["RF"] = rf

    # TABPFN
    if include_tabpfn:
        tp = TabPFNClassifier(device="auto")
        tp.fit(X_tr_sc, y_tr)
        models["TabPFN"] = tp

    return models


def proba_fn(model):
    return lambda X: model.predict_proba(X)[:, 1]


# ---- cell 8 (PATCHED: added "TabPFN" to the clone loop) ----
from sklearn.base import clone

def fit_models_for_dice(X_tr_raw_df, y_tr, pre, *, outcome_name="outcome", seed=0, models_sc=None):
    """
    Fit pipeline models for DiCE (pre -> estimator) on RAW training data.
    Uses clone(models_sc[name]) to preserve hyperparams.
    """
    X_tr_raw = X_tr_raw_df.drop(columns=[outcome_name], errors="ignore")

    dice_models = {}

    for name in ["LR", "XGB", "RF", "TabPFN"]:
        if models_sc is None or name not in models_sc:
            raise ValueError("Pass models_sc with fitted models to clone hyperparams cleanly.")
        est = clone(models_sc[name])   # keeps hyperparams, resets fitted state
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_tr_raw, y_tr)
        dice_models[name] = pipe

    return dice_models

# ---- cell 10 ----
def stable_int_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(s) & 0xFFFFFFFF  # 0..2^32-1

# ---- cell 11 ----
def l0_mixed(x_f, x_cf, feature_info, tol=1e-6):
    if feature_info is None or len(getattr(feature_info, "cat_groups", [])) == 0:
        return int(np.sum(np.abs(x_cf - x_f) > tol))

    d = x_f.shape[0]
    cat_groups = [np.asarray(g, int) for g in feature_info.cat_groups]

    if getattr(feature_info, "num_idx", None) is not None:
        num_idx = np.asarray(feature_info.num_idx, int)
    else:
        cat_cols = np.zeros(d, dtype=bool)
        for g in cat_groups:
            cat_cols[g] = True
        num_idx = np.flatnonzero(~cat_cols)

    num_changed = int(np.sum(np.abs(x_cf[num_idx] - x_f[num_idx]) > tol))

    cat_changed = 0
    for g in cat_groups:
        if np.any(np.abs(x_cf[g] - x_f[g]) > tol):
            cat_changed += 1

    return num_changed + cat_changed

# ---- cell 12 ----
def prepare_dataset(
    ds_in,
    *,
    rng=0,
    test_size=0.25,
    ohe_sparse=True,
    make_dense=True,
    outcome_name="outcome",          # <-- standardized target column name
    dice_clip_quantiles=None,        # e.g. (0.01, 0.99)
    dice_use_immutables=True,
):

    def dice_permitted_range_from_train(X_tr_raw_df, continuous_features, clip_quantiles=None):
        pr = {}
        for c in continuous_features:
            s = pd.to_numeric(X_tr_raw_df[c], errors="coerce")
            if s.isna().all():
                continue
            if clip_quantiles is None:
                lo = float(np.nanmin(s.values))
                hi = float(np.nanmax(s.values))
            else:
                qlo, qhi = clip_quantiles
                lo = float(np.nanquantile(s.values, qlo))
                hi = float(np.nanquantile(s.values, qhi))
            if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
                pr[c] = [lo, hi]
        return pr

    def dice_features_to_vary(num_cols, cat_cols, immutables=None, *, dice_use_immutables=True):
        """
        Return a list of feature names DiCE is allowed to vary.
        - num_cols, cat_cols: raw feature names (pre-OHE)
        - immutables: ImmutableSpec (names), or None
        """
        cols = list(num_cols) + list(cat_cols)

        if (not dice_use_immutables) or (immutables is None):
            return cols

        imm_num = set(immutables.num) & set(num_cols)
        imm_cat = set(immutables.cat) & set(cat_cols)
        imm = imm_num | imm_cat

        return [c for c in cols if c not in imm]


    # ---------------------------
    # REAL DATASET
    # ---------------------------
    if hasattr(ds_in, "X_df"):
        X_df = ds_in.X_df
        y = np.asarray(ds_in.y).astype(int, copy=False)
        num_cols, cat_cols = list(ds_in.num_cols), list(ds_in.cat_cols)

        idx = np.arange(len(X_df))
        idx_tr, idx_te = train_test_split(
            idx, test_size=test_size, stratify=y, random_state=rng
        )

        X_tr_raw_df = X_df.iloc[idx_tr].copy()
        X_te_raw_df = X_df.iloc[idx_te].copy()
        y_tr = y[idx_tr]
        y_te = y[idx_te]

        # Add standardized outcome column for DiCE
        X_tr_raw_df[outcome_name] = y_tr
        X_te_raw_df[outcome_name] = y_te

        # Fit preprocessing on train only (features only)
        pre = make_preprocessor(num_cols, cat_cols, ohe_sparse=ohe_sparse)
        pre.fit(X_tr_raw_df.drop(columns=[outcome_name]))

        X_tr_sc = pre.transform(X_tr_raw_df.drop(columns=[outcome_name]))
        X_te_sc = pre.transform(X_te_raw_df.drop(columns=[outcome_name]))

        if make_dense and hasattr(X_tr_sc, "toarray"):
            X_tr_sc = X_tr_sc.toarray()
            X_te_sc = X_te_sc.toarray()

        X_tr_sc = np.asarray(X_tr_sc, dtype=np.float64)
        X_te_sc = np.asarray(X_te_sc, dtype=np.float64)

        immutables = getattr(ds_in, "immutables", ImmutableSpec.empty())
        has_immutables = bool(immutables.num) or bool(immutables.cat)

        if len(cat_cols) == 0 and not has_immutables:
            feature_info = None
        else:
            num_idx, cat_groups = build_feature_info_from_pre(pre, num_cols, cat_cols)
            immutable_num = safe_immutable_num(num_cols, num_idx, ds_in.immutables.num)
            immutable_cat = safe_immutable_cat(cat_cols, ds_in.immutables.cat)
            feature_info = FeatureInfo(
                num_idx=num_idx,
                cat_groups=cat_groups,
                immutable_num=immutable_num,
                immutable_cat=immutable_cat,
            )

        # DiCE spec (raw space)
        continuous_features = list(num_cols)
        categorical_features = list(cat_cols)
        dice_spec = {
            "continuous_features": continuous_features,
            "categorical_features": categorical_features,
            "outcome_name": outcome_name,
            "permitted_range": dice_permitted_range_from_train(
                X_tr_raw_df, continuous_features, clip_quantiles=dice_clip_quantiles
            ),
        }

        immutables = getattr(ds_in, "immutables", ImmutableSpec.empty())

        dice_spec["features_to_vary"] = dice_features_to_vary(
            num_cols=num_cols,
            cat_cols=cat_cols,
            immutables=immutables,
            dice_use_immutables=True,
        )

        meta = dict(getattr(ds_in, "meta", {}) or {})
        meta.update({
            "n_rows": int(len(X_df)),
            "n_cols_model": int(X_tr_sc.shape[1]),
            "idx_tr": idx_tr,
            "idx_te": idx_te,
            "num_cols": continuous_features,
            "cat_cols": categorical_features,
            "outcome_name": outcome_name,
        })

        return X_tr_sc, X_te_sc, y_tr, y_te, feature_info, meta, pre, X_tr_raw_df, X_te_raw_df, dice_spec

    # ---------------------------
    # SYNTHETIC DATASET
    # ---------------------------
    X = np.asarray(ds_in.X, dtype=np.float64)
    y = np.asarray(ds_in.y, dtype=int)

    X_tr_sc, X_te_sc, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rng
    )

    d = X.shape[1]
    cols = [f"f{i}" for i in range(d)]
    X_tr_raw_df = pd.DataFrame(X_tr_sc, columns=cols)
    X_te_raw_df = pd.DataFrame(X_te_sc, columns=cols)

    X_tr_raw_df[outcome_name] = y_tr
    X_te_raw_df[outcome_name] = y_te

    feature_info = None
    pre = None

    dice_spec = {
        "continuous_features": cols,
        "categorical_features": [],
        "outcome_name": outcome_name,
        "features_to_vary": cols,
        "permitted_range": dice_permitted_range_from_train(
            X_tr_raw_df, cols, clip_quantiles=dice_clip_quantiles
        ),
    }

    meta = dict(getattr(ds_in, "meta", {}) or {})
    meta.update({
        "n_rows": int(X.shape[0]),
        "n_cols_model": int(d),
        "idx_tr": None,
        "idx_te": None,
        "num_cols": cols,
        "cat_cols": [],
        "outcome_name": outcome_name,
    })

    return X_tr_sc, X_te_sc, y_tr, y_te, feature_info, meta, pre, X_tr_raw_df, X_te_raw_df, dice_spec

# ---- cell 13 ----
def _mk_rep(ok, status, msg=None, **extra):
    rep = {"ok": bool(ok), "status": str(status)}
    if msg is not None:
        rep["msg"] = str(msg)
    rep.update(extra)
    return rep


def dice_to_standard_out(
    *,
    dice_exp,                 # object returned by dice.generate_counterfactuals(...)
    pre,                      # fitted preprocessor (ColumnTransformer / Pipeline step)
    x_f_raw_df,               # factual row as 1-row DF (features only, no outcome col)
    dice_spec,                # dict with outcome_name, etc.
    pick="closest_l2_scaled", # how to choose among multiple CFs
    make_dense=True,
):
    outcome_name = dice_spec["outcome_name"]

    try:
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
    except Exception as e:
        return None, _mk_rep(False, "dice_no_output", str(e)), {"dice_exp": dice_exp}

    if cf_df is None or len(cf_df) == 0:
        return None, _mk_rep(False, "dice_no_cfs"), {"cf_df": cf_df}

    cf_feat_df = cf_df.copy()
    if outcome_name in cf_feat_df.columns:
        cf_feat_df = cf_feat_df.drop(columns=[outcome_name])

    cf_feat_df = cf_feat_df[x_f_raw_df.columns]

    if pre is None:
        X_cf_sc = cf_feat_df.to_numpy(dtype=np.float64, copy=False)
    else:
        X_cf_sc = pre.transform(cf_feat_df)
        if make_dense and hasattr(X_cf_sc, "toarray"):
            X_cf_sc = X_cf_sc.toarray()
        X_cf_sc = X_cf_sc.astype(np.float64, copy=False)

    if make_dense and hasattr(X_cf_sc, "toarray"):
        X_cf_sc = X_cf_sc.toarray()

    X_cf_sc = np.asarray(X_cf_sc, dtype=np.float64)

    if pre is None:
        X_f_sc = x_f_raw_df.to_numpy(dtype=np.float64, copy=False).reshape(1, -1)
    else:
        X_f_sc = pre.transform(x_f_raw_df)

    if make_dense and hasattr(X_f_sc, "toarray"):
        X_f_sc = X_f_sc.toarray()
    X_f_sc = np.asarray(X_f_sc, dtype=np.float64).reshape(1, -1)

    if pick == "first":
        j = 0
        scores = None
    elif pick == "closest_l2_scaled":
        dif = X_cf_sc - X_f_sc
        scores = np.sqrt((dif * dif).sum(axis=1))
        j = int(np.argmin(scores))
    else:
        raise ValueError(f"Unknown pick policy: {pick}")

    x_cf_sc = X_cf_sc[j].copy()

    meta = {
        "n_cfs_returned": int(len(cf_df)),
        "chosen_idx": int(j),
        "pick_policy": pick,
        "cf_raw_df": cf_df,
        "cf_features_df": cf_feat_df,
    }
    if scores is not None:
        meta["pick_scores"] = scores

    rep = _mk_rep(True, "dice_ok", chosen_idx=j, n_cfs=len(cf_df))

    return x_cf_sc, rep, meta

# ---- cell 14 ----
def validate_cf(
    *,
    x_cf_sc,
    x_f_sc,
    predict_fn_proba,
    proba_threshold=0.5,
    tol=1e-8,
):
    info = {}

    if x_cf_sc is None:
        return False, "no_cf", info

    x_cf_sc = np.asarray(x_cf_sc)

    if x_cf_sc.ndim != 1:
        return False, "bad_shape", {"shape": x_cf_sc.shape}

    if not np.all(np.isfinite(x_cf_sc)):
        return False, "non_finite", {}

    delta = x_cf_sc - x_f_sc
    l0 = int(np.sum(np.abs(delta) > tol))
    info["l0"] = l0

    if l0 == 0:
        return False, "no_change", info

    p_cf = predict_fn_proba(x_cf_sc.reshape(1, -1))[0]
    p_f  = predict_fn_proba(x_f_sc.reshape(1, -1))[0]

    info["proba_f"] = p_f
    info["proba_cf"] = p_cf

    p1_f  = float(p_f[1]) if len(p_f) == 2 else float(p_f[0])
    p1_cf = float(p_cf[1]) if len(p_cf) == 2 else float(p_cf[0])

    flipped = (p1_f < proba_threshold) and (p1_cf >= proba_threshold)
    flipped |= (p1_f >= proba_threshold) and (p1_cf < proba_threshold)

    if not flipped:
        return False, "no_flip", info

    info["l2"] = float(np.linalg.norm(delta))

    return True, "ok", info

# ---- cell 15 (PATCHED: dice_exp always rebuilt per model_name + assert guard) ----
def run_benchmark(
    *,
    X_tr_sc, X_te_sc, y_tr, y_te,
    models: dict,
    factual_indices: np.ndarray,
    methods=("pace", "nice", "dice"),
    seed=0,
    guided_n_candidates=1000,
    guided_max_changed_features=8,
    guided_anchor_k=250,
    guided_stageA_random_state=0,
    guided_anchors_random_state=1,
    feature_info=None,
    pre=None,
    X_tr_raw=None,
    X_te_raw=None,
    dice_spec=None,
    include_vectors: bool = True,
    max_vec_len: int | None = None,
    dataset_name: str = "",
):
    def _maybe_trunc(v: np.ndarray) -> list:
        v = np.asarray(v).reshape(-1)
        if (max_vec_len is not None) and (v.size > max_vec_len):
            return v[:max_vec_len].tolist()
        return v.tolist()

    rows = []
    best_sources = []
    statuses = []

    weights = DICE_WEIGHT_POLICIES.get(dice_spec["opt_method"], {})

    for model_name, model in models.items():
        print("Working with model", model_name, flush=True)
        pp = proba_fn(model)
        pp_guidance = proba_fn(models["LR"])
        pp_target = proba_fn(model)

        p_te = pp(X_te_sc)
        acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))
        auc = roc_auc_score(y_te, p_te)

        predict_fn_proba = lambda X: model.predict_proba(X)  # 2D proba
        nice_base_obj = nice_base_ctx = None
        nice_spars_obj = nice_spars_ctx = None
        mcce_obj = mcce_ctx = None

        if "dice" in methods:
            dice_data = dice_ml.Data(
                dataframe=X_tr_raw,
                continuous_features=dice_spec["continuous_features"],
                categorical_features=dice_spec["categorical_features"],
                outcome_name=dice_spec["outcome_name"]
            )
            dice_model = dice_ml.Model(model=dice_spec['models'][model_name], backend="sklearn")
            dice_exp = Dice(dice_data, dice_model, method=dice_spec['opt_method'])
            assert dice_exp.model.model is dice_spec['models'][model_name], (
                f"dice_exp built for wrong model: expected {model_name}"
            )

        guided_cache = None
        if "pace" in methods:
            guided_cache = guided_cf_build_cache(
                X_tr_sc=X_tr_sc,
                predict_proba_target=pp_target,
                predict_proba_guidance=pp_guidance,
                stageA_random_state=guided_stageA_random_state,
                anchor_k=guided_anchor_k,
                anchors_random_state=guided_anchors_random_state,
                feature_info=feature_info,
            )

        for idx in factual_indices:
            x_f = X_te_sc[idx]
            p_f = float(pp(x_f[None, :])[0])
            yhat_f = int(p_f >= 0.5)
            y_desired = 1 - yhat_f

            for method in methods:
                t0 = time.perf_counter()
                if method == "pace":
                    guided_rs = stable_int_seed(seed, model_name, "guided", int(idx))
                    x_cf, rep, meta, C = guided_cf_one_repeated(
                        x_f_sc=x_f,
                        cache=guided_cache,
                        predict_proba_target=pp,
                        n_candidates=guided_n_candidates,
                        max_changed_features=guided_max_changed_features,
                        random_state=guided_rs,
                        feature_info=feature_info,
                    )
                    statuses.append(rep.get("status", ""))
                    if rep.get("status") == "ok":
                        best_sources.append(rep["best_source"])

                elif method == "nice_base":
                    if nice_base_obj is None:
                        continue
                    out = nice_cf_one(
                        nice_obj=nice_base_obj, x_f_sc=x_f, predict_proba_target=pp,
                        proba_threshold=0.5, tol=1e-6, feature_info=feature_info,
                        repair_onehot=False, nice_ctx=nice_base_ctx,
                    )
                    x_cf, rep, meta = out.x_cf, out.rep, out.meta

                elif method == "nice_spars":
                    if nice_spars_obj is None:
                        continue
                    out = nice_cf_one(
                        nice_obj=nice_spars_obj, x_f_sc=x_f, predict_proba_target=pp,
                        proba_threshold=0.5, tol=1e-6, feature_info=feature_info,
                        repair_onehot=False, nice_ctx=nice_spars_ctx,
                    )
                    x_cf, rep, meta = out.x_cf, out.rep, out.meta

                elif method == "mcce":
                    if mcce_obj is None:
                        continue
                    out = mcce_cf_one(
                        mcce_obj=mcce_obj, mcce_ctx=mcce_ctx, x_f_sc=x_f,
                        y_desired=y_desired, model=model, proba_threshold=0.5, k=1000,
                    )
                    x_cf, rep, meta = out.x_cf, out.rep, out.meta

                elif method == 'dice':
                    x_f_dice = X_te_raw.drop(columns=[dice_spec["outcome_name"]]).iloc[[idx]]
                    try:
                        x_cf_dice = dice_exp.generate_counterfactuals(
                            query_instances=x_f_dice,
                            total_CFs=1,
                            desired_class="opposite",
                            features_to_vary=dice_spec["features_to_vary"],
                            permitted_range=dice_spec["permitted_range"],
                            verbose=False,
                            **weights,
                        )
                    except UserConfigValidationException as e:
                        x_cf_dice = None
                        rep = {"ok": False, "status": "dice_invalid_config", "msg": str(e)}
                        meta = {"exception": "UserConfigValidationException", "exception_msg": str(e)}
                    except Exception as e:
                        x_cf_dice = None
                        rep = {"ok": False, "status": "dice_exception", "msg": str(e)}
                        meta = {"exception": type(e).__name__, "exception_msg": str(e)}

                    if x_cf_dice is not None:
                        x_cf_sc, rep, meta = dice_to_standard_out(
                            dice_exp=x_cf_dice, pre=pre, x_f_raw_df=x_f_dice,
                            dice_spec=dice_spec, pick="closest_l2_scaled", make_dense=True,
                        )
                        if x_cf_sc is not None:
                            _, status, _ = validate_cf(
                                x_cf_sc=x_cf_sc, x_f_sc=x_f.ravel(),
                                predict_fn_proba=predict_fn_proba, proba_threshold=0.5,
                            )
                            rep['status'] = status
                    else:
                        x_cf_sc = None
                    x_cf = x_cf_sc
                    out = SimpleNamespace(x_cf=x_cf_sc, rep=rep, meta=meta)

                else:
                    raise ValueError(f"Unknown method: {method}")

                dt = time.perf_counter() - t0
                ok = (x_cf is not None) and (rep.get("status") == "ok")

                if ok:
                    p_cf = float(pp(x_cf[None, :])[0])
                    flip_ok = int((p_cf >= 0.5) == y_desired)
                    diff = x_cf - x_f
                    l0 = l0_mixed(x_f, x_cf, feature_info, tol=1e-6)
                    l2 = float(np.sqrt(np.sum(diff * diff)))
                    margin = abs(p_cf - 0.5)
                    changed_idx = np.flatnonzero(np.abs(diff) > 1e-6)
                    changed_idx_show = changed_idx[:50].tolist()
                else:
                    p_cf = np.nan
                    flip_ok = 0
                    l0, l2, margin = np.nan, np.nan, np.nan
                    diff = None
                    changed_idx_show = []

                row = {
                    "model": model_name, "method": method, "idx": int(idx),
                    "ok": bool(ok), "flip_ok": int(flip_ok), "p_f": float(p_f),
                    "p_cf": float(p_cf) if np.isfinite(p_cf) else np.nan,
                    "margin_cf": float(margin) if np.isfinite(margin) else np.nan,
                    "l0": l0, "l2": l2, "time_s": float(dt),
                    "clf_acc": float(acc), "clf_auc": float(auc),
                    "status": rep.get("status"), "rep": rep,
                    "changed_idx": changed_idx_show,
                }

                if include_vectors:
                    row["x_f"] = _maybe_trunc(x_f)
                    row["x_cf"] = _maybe_trunc(x_cf) if ok else None
                    row["diff"] = _maybe_trunc(diff) if ok else None

                print(f"  [{model_name}/{method}] idx={idx} status={row['status']!r} "
                      f"ok={row['ok']} l0={row['l0']} l2={row['l2']:.4f} "
                      f"time_s={row['time_s']:.2f}", flush=True)

                rows.append(row)

    return pd.DataFrame(rows), statuses, best_sources

# ---- cell 17 ----
def choose_dice_method(dataset_name: str) -> str:
    if dataset_name.startswith("mc_") or dataset_name.startswith("synthetic"):
        return "genetic"
    return "random"

# ---- cell 18 ----
DICE_WEIGHT_POLICIES = {
    "random": {},
    "genetic": dict(proximity_weight=1.0, sparsity_weight=1.0, diversity_weight=0.0),
}

DICE_CLIP_QUANTILES = {
    "ilpd": (0.01, 0.99),
}



# ==========================================================================
# Shared helper for driver scripts: force a particular DiCE opt_method
# regardless of what choose_dice_method(dataset_name) would normally pick.
# Used to test genetic-on-real-world and to make genetic-on-synthetic explicit.
# ==========================================================================
def force_opt_method(dice_spec, opt_method):
    dice_spec['opt_method'] = opt_method
    return dice_spec


# ---- cell 6 (added for NICE-sparse TabPFN evidence tests) ----
def build_nice_explainer(
    *,
    X_train_sc: np.ndarray,
    y_train: np.ndarray,
    predict_fn_proba: Callable[[np.ndarray], np.ndarray],
    feature_info: Optional[FeatureInfo] = None,
    nice_opt: str = 'none',
    # Raw-space arguments for proper categorical handling
    X_tr_raw_df=None,   # raw training features WITHOUT outcome column
    model=None,          # fitted classifier
    pre=None,            # fitted ColumnTransformer
    num_cols=None,       # list of numeric column names
    cat_cols=None,       # list of categorical column names
    max_train: Optional[int] = None,  # None = auto-detect per model type
):
    """
    Build NICE object once. Reuse it across factual points.

    When X_tr_raw_df + model + pre + num_cols + cat_cols are supplied AND cat_cols is
    non-empty, NICE operates in the original (pre-OHE) feature space with ordinal-encoded
    categoricals.  This is the correct usage of NICE and avoids the distortion caused by
    treating individual one-hot dummy columns as independent categorical features.

    Otherwise falls back to OHE-space behaviour (used for num-only / synthetic datasets).
    """
    from nice import NICE
    from sklearn.preprocessing import OrdinalEncoder

    # Auto-cap for TabPFN: its per-call inference cost makes scanning large training sets infeasible.
    if max_train is None and model is not None:
        try:
            from tabpfn import TabPFNClassifier
            if isinstance(model, TabPFNClassifier):
                max_train = 5000
        except ImportError:
            pass

    # Stratified subsample of the neighbour pool when needed.
    y_train_arr = np.asarray(y_train)
    if max_train is not None and len(y_train_arr) > max_train:
        rng_sub = np.random.default_rng(0)
        classes, counts = np.unique(y_train_arr, return_counts=True)
        idx_sub = []
        for cls, cnt in zip(classes, counts):
            cls_idx = np.where(y_train_arr == cls)[0]
            n_cls = max(1, round(max_train * int(cnt) / len(y_train_arr)))
            idx_sub.append(rng_sub.choice(cls_idx, min(n_cls, len(cls_idx)), replace=False))
        idx_sub = np.concatenate(idx_sub)
        rng_sub.shuffle(idx_sub)
        X_train_sc = X_train_sc[idx_sub]
        y_train = y_train_arr[idx_sub]
        if X_tr_raw_df is not None:
            X_tr_raw_df = X_tr_raw_df.iloc[idx_sub].reset_index(drop=True)

    use_raw_space = (
        X_tr_raw_df is not None
        and cat_cols is not None
        and len(cat_cols) > 0
        and model is not None
        and pre is not None
    )

    if use_raw_space:
        all_cols = list(num_cols) + list(cat_cols)
        X_raw = X_tr_raw_df[all_cols].copy()

        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                            dtype=np.float64)
        X_raw[cat_cols] = oe.fit_transform(X_raw[cat_cols])
        X_nice = X_raw[all_cols].to_numpy(dtype=np.float64)

        n_num = len(num_cols)
        num_feat = list(range(n_num))
        cat_feat = list(range(n_num, len(all_cols)))

        # Capture in closure to avoid late-binding issues
        _pre, _model, _oe = pre, model, oe
        _num_cols, _cat_cols, _all_cols = list(num_cols), list(cat_cols), all_cols

        def _predict_fn(X_arr):
            df = pd.DataFrame(X_arr.astype(object), columns=_all_cols)
            df[_cat_cols] = _oe.inverse_transform(
                np.asarray(X_arr[:, n_num:], dtype=np.float64)
            )
            for c in _num_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            Xsc = _pre.transform(df)
            if hasattr(Xsc, 'toarray'):
                Xsc = Xsc.toarray()
            return _model.predict_proba(np.asarray(Xsc, dtype=np.float64))

        nice_obj = NICE(
            X_train=X_nice,
            predict_fn=_predict_fn,
            y_train=y_train,
            cat_feat=cat_feat,
            num_feat=num_feat,
            distance_metric='HEOM',
            num_normalization='minmax',
            optimization=nice_opt,
            justified_cf=True,
        )
        nice_ctx = SimpleNamespace(
            use_raw_space=True,
            oe=oe,
            all_cols=all_cols,
            num_cols=list(num_cols),
            cat_cols=list(cat_cols),
            pre=pre,
        )
        return nice_obj, nice_ctx

    # ---- fallback: OHE-space (num-only or synthetic datasets) ----
    d = X_train_sc.shape[1]

    if feature_info is not None and len(getattr(feature_info, "cat_groups", [])) > 0:
        cat_cols_sc = np.concatenate([np.asarray(g, int) for g in feature_info.cat_groups]).tolist()
        if feature_info.num_idx is not None:
            num_feat = np.asarray(feature_info.num_idx, int).tolist()
        else:
            mask = np.ones(d, dtype=bool)
            mask[np.asarray(cat_cols_sc, int)] = False
            num_feat = np.flatnonzero(mask).tolist()
        cat_feat = cat_cols_sc
    else:
        cat_feat = []
        num_feat = list(range(d))

    nice_obj = NICE(
        X_train=X_train_sc,
        predict_fn=predict_fn_proba,
        y_train=y_train,
        cat_feat=cat_feat,
        num_feat=num_feat,
        distance_metric="HEOM",
        num_normalization="minmax",
        optimization=nice_opt,
        justified_cf=True,
    )
    nice_ctx = SimpleNamespace(use_raw_space=False)
    return nice_obj, nice_ctx


def _sc_to_nice_vec(x_sc: np.ndarray, nice_ctx) -> np.ndarray:
    """Convert a model-space (scaled/OHE) vector to NICE's ordinal raw space."""
    raw = inverse_preprocess(x_sc, nice_ctx.pre)
    row = []
    for c in nice_ctx.all_cols:
        if c in nice_ctx.num_cols:
            row.append(float(raw.get(c, np.nan)))
        else:
            row.append(raw.get(c, None))
    df = pd.DataFrame([row], columns=nice_ctx.all_cols)
    df[nice_ctx.cat_cols] = nice_ctx.oe.transform(df[nice_ctx.cat_cols])
    for c in nice_ctx.num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df[nice_ctx.all_cols].to_numpy(dtype=np.float64).ravel()


def _nice_vec_to_sc(x_nice: np.ndarray, nice_ctx) -> np.ndarray:
    """Convert a NICE ordinal-space vector back to model space (scaled/OHE)."""
    n_num = len(nice_ctx.num_cols)
    df = pd.DataFrame([x_nice], columns=nice_ctx.all_cols, dtype=object)
    df[nice_ctx.cat_cols] = nice_ctx.oe.inverse_transform(
        np.asarray([x_nice[n_num:]], dtype=np.float64)
    )
    for c in nice_ctx.num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    Xsc = nice_ctx.pre.transform(df)
    if hasattr(Xsc, 'toarray'):
        Xsc = Xsc.toarray()
    return np.asarray(Xsc, dtype=np.float64).ravel()


@dataclass
class CFOut:
    x_cf: Optional[np.ndarray]
    rep: Dict[str, Any]
    meta: Dict[str, Any]


def _extract_cf_from_nice_result(res) -> Optional[np.ndarray]:
    """Tries hard to extract a CF vector from whatever NICE returns."""
    if res is None:
        return None
    if isinstance(res, np.ndarray):
        if res.ndim == 2 and res.shape[0] == 1:
            return res[0]
        if res.ndim == 1:
            return res
    return None


def nice_cf_one(
    *,
    nice_obj,
    x_f_sc: np.ndarray,
    predict_proba_target: PredictProbaFn,
    proba_threshold: float = 0.5,
    tol: float = 1e-6,
    feature_info: Optional["FeatureInfo"] = None,
    repair_onehot: bool = False,
    nice_ctx=None,   # SimpleNamespace from build_nice_explainer
) -> CFOut:
    """
    Run NICE for one factual and return a standardized output.

    If nice_ctx.use_raw_space is True, the factual is converted to ordinal space before
    calling NICE and the CF is converted back to model space for metric computation.
    This is the correct workflow for datasets with categorical features.
    """
    t0 = time.perf_counter()

    x_f_sc = np.asarray(x_f_sc, dtype=np.float32).reshape(-1)
    p_f = float(predict_proba_target(x_f_sc[None, :])[0])
    yhat_f = int(p_f >= proba_threshold)
    y_desired = 1 - yhat_f

    # ---- raw-space path (proper categorical handling) ----
    if nice_ctx is not None and nice_ctx.use_raw_space:
        try:
            x_f_nice = _sc_to_nice_vec(x_f_sc, nice_ctx)
            res = nice_obj.explain(x_f_nice[None, :])
        except Exception as e:
            return CFOut(
                x_cf=None,
                rep={"status": "exception", "exception": repr(e)},
                meta={"method": "NICE"},
            )

        x_cf_nice = _extract_cf_from_nice_result(res)
        if x_cf_nice is None:
            return CFOut(
                x_cf=None,
                rep={"status": "no_cf"},
                meta={"method": "NICE", "raw": res},
            )

        # Post-hoc immutability check in ordinal space (before back-converting)
        if feature_info is not None:
            _n_num_rc = len(nice_ctx.num_cols)
            for _j in getattr(feature_info, "immutable_num", frozenset()):
                if abs(float(x_cf_nice[int(_j)]) - float(x_f_nice[int(_j)])) > tol:
                    dt = time.perf_counter() - t0
                    return CFOut(
                        x_cf=None,
                        rep={"status": "violates_immutable", "feature": int(_j), "time_s": float(dt)},
                        meta={"method": "NICE"},
                    )
            for _k in getattr(feature_info, "immutable_cat", frozenset()):
                _pos = _n_num_rc + int(_k)
                if abs(float(x_cf_nice[_pos]) - float(x_f_nice[_pos])) > tol:
                    dt = time.perf_counter() - t0
                    return CFOut(
                        x_cf=None,
                        rep={"status": "violates_immutable", "feature": int(_k), "time_s": float(dt)},
                        meta={"method": "NICE"},
                    )

        try:
            x_cf = _nice_vec_to_sc(x_cf_nice, nice_ctx)
        except Exception as e:
            return CFOut(
                x_cf=None,
                rep={"status": "decode_error", "exception": repr(e)},
                meta={"method": "NICE"},
            )

        x_f_sc_f64 = x_f_sc.astype(np.float64)
        p_cf = float(predict_proba_target(x_cf[None, :])[0])
        yhat_cf = int(p_cf >= proba_threshold)
        dt = time.perf_counter() - t0

        if yhat_cf != y_desired:
            return CFOut(
                x_cf=x_cf,
                rep={
                    "status": "not_flipped",
                    "p_f": float(p_f), "p_cf": float(p_cf),
                    "yhat_f": int(yhat_f), "y_desired": int(y_desired),
                    "yhat_cf": int(yhat_cf), "time_s": float(dt),
                },
                meta={"method": "NICE", "raw": res},
            )

        diff = x_cf - x_f_sc_f64
        if feature_info is not None and len(getattr(feature_info, "cat_groups", [])) > 0:
            cat_groups = [np.asarray(g, int) for g in feature_info.cat_groups]
            if getattr(feature_info, "num_idx", None) is not None:
                num_idx = np.asarray(feature_info.num_idx, int)
            else:
                all_cat = np.concatenate(cat_groups)
                num_idx = np.flatnonzero(~np.isin(np.arange(len(x_f_sc)), all_cat))
            l0 = int(l0_group_aware(x_f_sc_f64, x_cf[None, :],
                                     num_idx=num_idx, cat_groups=cat_groups, tol=tol)[0])
        else:
            l0 = int(np.sum(np.abs(diff) > tol))
        l2 = float(np.sqrt(np.sum(diff * diff)))

        return CFOut(
            x_cf=x_cf,
            rep={
                "status": "ok",
                "p_f": float(p_f), "p_cf": float(p_cf),
                "yhat_f": int(yhat_f), "y_desired": int(y_desired),
                "l0": int(l0), "l2": float(l2), "time_s": float(dt),
            },
            meta={"method": "NICE", "raw": res, "raw_space": True},
        )

    # ---- OHE-space path (fallback for num-only / synthetic datasets) ----
    try:
        res = nice_obj.explain(x_f_sc[None, :])
    except Exception as e:
        return CFOut(
            x_cf=None,
            rep={"status": "exception", "exception": repr(e)},
            meta={"method": "NICE"},
        )

    x_cf = _extract_cf_from_nice_result(res)
    if x_cf is None:
        return CFOut(
            x_cf=None,
            rep={"status": "no_cf"},
            meta={"method": "NICE", "raw": res},
        )

    x_cf = np.asarray(x_cf, dtype=np.float32).reshape(-1)

    if feature_info is not None and len(getattr(feature_info, "cat_groups", [])) > 0:
        cat_groups = [np.asarray(g, dtype=int) for g in feature_info.cat_groups]

        if repair_onehot:
            x_cf = x_cf.copy()
            for g in cat_groups:
                j = int(g[np.argmax(x_cf[g])])
                x_cf[g] = 0.0
                x_cf[j] = 1.0
        else:
            if not onehot_valid_mask(x_cf[None, :], cat_groups, tol=tol)[0]:
                dt = time.perf_counter() - t0
                return CFOut(
                    x_cf=x_cf,
                    rep={"status": "invalid_onehot", "time_s": float(dt)},
                    meta={"method": "NICE", "raw": res},
                )

        immutable_num = getattr(feature_info, "immutable_num", frozenset())
        if immutable_num:
            for j in immutable_num:
                if abs(float(x_cf[int(j)] - x_f_sc[int(j)])) > tol:
                    dt = time.perf_counter() - t0
                    return CFOut(
                        x_cf=x_cf,
                        rep={"status": "violates_immutable", "feature": int(j), "time_s": float(dt)},
                        meta={"method": "NICE", "raw": res},
                    )
        immutable_cat = getattr(feature_info, "immutable_cat", frozenset())
        if immutable_cat:
            _cg = [np.asarray(g, int) for g in feature_info.cat_groups]
            for k in immutable_cat:
                g = _cg[int(k)]
                if np.any(np.abs(x_cf[g] - x_f_sc[g]) > tol):
                    dt = time.perf_counter() - t0
                    return CFOut(
                        x_cf=x_cf,
                        rep={"status": "violates_immutable", "feature": int(k), "time_s": float(dt)},
                        meta={"method": "NICE", "raw": res},
                    )

    p_cf = float(predict_proba_target(x_cf[None, :])[0])
    yhat_cf = int(p_cf >= proba_threshold)

    if yhat_cf != y_desired:
        dt = time.perf_counter() - t0
        return CFOut(
            x_cf=x_cf,
            rep={
                "status": "not_flipped",
                "p_f": float(p_f), "p_cf": float(p_cf),
                "yhat_f": int(yhat_f), "y_desired": int(y_desired),
                "yhat_cf": int(yhat_cf), "time_s": float(dt),
            },
            meta={"method": "NICE", "raw": res},
        )

    diff = x_cf - x_f_sc

    if feature_info is not None and len(getattr(feature_info, "cat_groups", [])) > 0:
        d = x_f_sc.shape[0]
        cat_groups = [np.asarray(g, int) for g in feature_info.cat_groups]
        if getattr(feature_info, "num_idx", None) is not None:
            num_idx = np.asarray(feature_info.num_idx, int)
        else:
            cat_cols_mask = np.zeros(d, dtype=bool)
            for g in cat_groups:
                cat_cols_mask[g] = True
            num_idx = np.flatnonzero(~cat_cols_mask)
        l0 = int(l0_group_aware(x_f_sc, x_cf[None, :], num_idx=num_idx,
                                  cat_groups=cat_groups, tol=tol)[0])
    else:
        l0 = int(np.sum(np.abs(diff) > tol))

    l2 = float(np.sqrt(np.sum(diff * diff)))
    dt = time.perf_counter() - t0

    rep = {
        "status": "ok",
        "p_f": float(p_f), "p_cf": float(p_cf),
        "yhat_f": int(yhat_f), "y_desired": int(y_desired),
        "l0": int(l0), "l2": float(l2), "time_s": float(dt),
    }
    meta = {"method": "NICE", "raw": res, "repaired_onehot": bool(repair_onehot)}
    return CFOut(x_cf=x_cf, rep=rep, meta=meta)
