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
# Driver: small bounded feasibility/timing test
# ==========================================================================
if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    N_FACTUAL = 5
    SEED = 0
    DATASET = "blood_transfusion"

    t_start = time.perf_counter()

    real_datasets = load_many([DATASET], openml_version=2)
    ds_in = real_datasets[0]
    print(f"Loaded dataset {ds_in.name!r}", flush=True)

    clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=SEED, dice_clip_quantiles=clip_q)

    print("Fitting LR/XGB/RF/TabPFN...", flush=True)
    t0 = time.perf_counter()
    models = fit_models(X_tr, y_tr, seed=SEED, include_tabpfn=True)
    print(f"  fit_models took {time.perf_counter() - t0:.1f}s", flush=True)

    print("Fitting DiCE pipelines (LR/XGB/RF/TabPFN)...", flush=True)
    t0 = time.perf_counter()
    dice_spec['models'] = fit_models_for_dice(
        X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=SEED, models_sc=models
    )
    print(f"  fit_models_for_dice took {time.perf_counter() - t0:.1f}s", flush=True)

    dice_spec['opt_method'] = choose_dice_method(ds_in.name)
    print(f"DiCE opt_method for {ds_in.name!r}: {dice_spec['opt_method']!r}", flush=True)

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)
    print(f"Factual indices: {factual_indices.tolist()}", flush=True)

    # Only LR (needed internally for pp_guidance) + TabPFN (the model under test)
    test_models = {"LR": models["LR"], "TabPFN": models["TabPFN"]}

    print("Running DiCE against LR and TabPFN...", flush=True)
    df, _, _ = run_benchmark(
        X_tr_sc=X_tr, X_te_sc=X_te, y_tr=y_tr, y_te=y_te,
        models=test_models,
        factual_indices=factual_indices,
        methods=("dice",),
        seed=SEED,
        feature_info=feature_info,
        pre=pre,
        X_tr_raw=X_tr_raw,
        X_te_raw=X_te_raw,
        dice_spec=dice_spec,
        include_vectors=False,
        dataset_name=ds_in.name,
    )

    t_total = time.perf_counter() - t_start

    print("\n===== SUMMARY =====", flush=True)
    print(df[["model", "idx", "status", "ok", "l0", "l2", "time_s"]].to_string(index=False), flush=True)
    print(flush=True)
    summary = df.groupby("model").agg(
        n=("ok", "size"),
        n_ok=("ok", "sum"),
        mean_time_s=("time_s", "mean"),
        median_time_s=("time_s", "median"),
        max_time_s=("time_s", "max"),
    )
    print(summary, flush=True)
    print(f"\nTotal wall clock for this script: {t_total:.1f}s", flush=True)

    out_path = "/nvme/h/lchristodoulou/pace/cf-proposal/e5b_test_dice_tabpfn_result.csv"
    df.drop(columns=["rep", "changed_idx"], errors="ignore").to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)
