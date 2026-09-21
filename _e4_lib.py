# ==========================================================================
# E4: instrument target-model cost to demonstrate (not assert) PACE's
# efficiency mechanism, and give E1b a call-count baseline for the declined
# MOC benchmark.
#
# `guided_cf_build_cache` / `guided_cf_one_repeated` / `GuidedCFCache` are
# extracted verbatim from tabpfn_cf5.ipynb (the cell immediately before
# run_benchmark) -- they are the actual PACE call path used to produce the
# paper's results, but were never pulled into _e5_lib.py (that extraction
# only covered what E5b's DiCE fix needed). Reproduced here rather than
# added to _e5_lib.py to avoid touching a file other regen scripts already
# depend on. The only change from the notebook source: `feature_info`'s
# default was the literal expression `FeatureInfo | None` (a latent bug --
# a runtime UnionType object, not None); every call site passes
# feature_info explicitly so this was never hit, but it's written correctly
# here since this is new code, not a preserved historical artifact.
#
# `run_benchmark_instrumented` is _e5_lib.py's run_benchmark (itself copied
# from tabpfn_cf5.ipynb cell 15, post-E5b-fix) with target-model call
# counting added. The trimmed _e5_lib.run_benchmark dropped the nice_base/
# nice_spars/mcce build blocks (dead code there -- always None -- since the
# E5b task only exercised "dice"); this restores them from the same
# notebook cell so all five methods are actually runnable.
#
# Instrumentation design: wrap only the *generation/search* call for each
# method (guided_cf_one_repeated / nice_cf_one / mcce_cf_one / DiCE's
# generate_counterfactuals), not the harness-level post-hoc validation
# calls that run uniformly across every method (the final `pp(x_cf)`
# metric recomputation after the if/elif chain, and -- for DiCE
# specifically -- dice_to_standard_out/validate_cf's re-check on the plain
# `model` after generate_counterfactuals returns). Those are constant
# across methods and would just add uninformative noise to a cost
# comparison that's about each method's own search process.
# ==========================================================================

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import time

from _e5_lib import *  # noqa: F401,F403 -- prepare_dataset, fit_models,
                        # fit_models_for_dice, proba_fn, stable_int_seed,
                        # l0_mixed, dice_to_standard_out, validate_cf,
                        # build_nice_explainer, nice_cf_one, choose_dice_method,
                        # DICE_WEIGHT_POLICIES, DICE_CLIP_QUANTILES,
                        # force_opt_method, UserConfigValidationException, and
                        # pfn_cf_guided_onehot's stageA_build /
                        # stageA_target_anchors_from_p / stageB_generate /
                        # stageC_select_best / FeatureInfo (via _e5_lib's own
                        # `from pfn_cf_guided_onehot import *`)

from mcce_cf import build_mcce_explainer, mcce_cf_one


# ---------------------------------------------------------------------------
# Call-counting instrumentation
# ---------------------------------------------------------------------------

class CallCounter:
    """Accumulates calls/rows for one patched predict_proba."""

    def __init__(self):
        self.n_calls = 0
        self.n_rows = 0
        self.per_call_rows = []

    def record(self, n_rows: int):
        self.n_calls += 1
        self.n_rows += int(n_rows)
        self.per_call_rows.append(int(n_rows))


@contextmanager
def count_predict_proba(obj):
    """
    Temporarily monkeypatches obj.predict_proba to count calls and total
    rows scored, then restores the original method exactly (instance dict
    put back to its prior state, whether or not it had its own override
    already).
    """
    counter = CallCounter()
    had_own = "predict_proba" in obj.__dict__
    prior_value = obj.__dict__.get("predict_proba", None)
    bound_original = obj.predict_proba  # resolves via MRO if not own

    def wrapped(X, *a, **kw):
        X_arr = np.asarray(X)
        n_rows = X_arr.shape[0] if X_arr.ndim >= 1 else 1
        counter.record(n_rows)
        return bound_original(X, *a, **kw)

    obj.predict_proba = wrapped
    try:
        yield counter
    finally:
        if had_own:
            obj.__dict__["predict_proba"] = prior_value
        else:
            del obj.__dict__["predict_proba"]


# ---------------------------------------------------------------------------
# PACE cache/call path (verbatim from tabpfn_cf5.ipynb; missing from _e5_lib.py)
# ---------------------------------------------------------------------------

@dataclass
class GuidedCFCache:
    X_tr_sc: np.ndarray
    predict_proba_guidance: Any
    feature_info: Optional[FeatureInfo]
    p_train: np.ndarray
    A: Any
    anchors_by_class: Dict[int, np.ndarray]
    anchors_diag_by_class: Dict[int, Dict[str, Any]]


def guided_cf_build_cache(
    *,
    X_tr_sc: np.ndarray,
    predict_proba_target,
    predict_proba_guidance=None,
    stageA_random_state: int = 0,
    anchor_k: int = 200,
    anchors_random_state: int = 1,
    feature_info: Optional[FeatureInfo] = None,
) -> GuidedCFCache:
    if predict_proba_guidance is None:
        predict_proba_guidance = predict_proba_target

    p_train = predict_proba_guidance(X_tr_sc).astype(np.float64)

    A = stageA_build(
        X_train=X_tr_sc,
        predict_proba_fn=predict_proba_guidance,
        random_state=stageA_random_state,
        feature_info=feature_info,
    )

    anchors_by_class = {}
    anchors_diag_by_class = {}

    for y_desired in (0, 1):
        anchors, diag = stageA_target_anchors_from_p(
            p_train=p_train,
            y_desired=y_desired,
            anchor_k=anchor_k,
            random_state=anchors_random_state,
        )
        anchors_by_class[y_desired] = anchors
        anchors_diag_by_class[y_desired] = diag

    return GuidedCFCache(
        X_tr_sc=X_tr_sc,
        predict_proba_guidance=predict_proba_guidance,
        feature_info=feature_info,
        p_train=p_train,
        A=A,
        anchors_by_class=anchors_by_class,
        anchors_diag_by_class=anchors_diag_by_class,
    )


def guided_cf_one_repeated(
    *,
    x_f_sc: np.ndarray,
    cache: GuidedCFCache,
    predict_proba_target,
    n_candidates: int = 300,
    max_changed_features: int = 3,
    random_state=0,
    feature_info: Optional[FeatureInfo] = None,
):
    p_f = float(predict_proba_target(x_f_sc[None, :])[0])
    yhat_f = int(p_f >= 0.5)
    y_desired = 1 - yhat_f

    anchors = cache.anchors_by_class[y_desired]
    if anchors.size == 0:
        return x_f_sc.copy(), {"ok": False}, {
            "error": "no_anchors",
            "y_desired": y_desired,
            "p_f": p_f,
        }, None

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
        random_state=random_state,
    )

    x_cf_sc, rep = stageC_select_best(
        x_f=x_f_sc,
        C=C_sc,
        sources=meta["sources"],
        predict_proba_fn=predict_proba_target,
        y_desired=y_desired,
        feature_info=cache.feature_info,
    )

    meta = dict(meta)
    meta["anchors_diag"] = cache.anchors_diag_by_class[y_desired]
    meta["p_f"] = p_f
    meta["y_desired"] = y_desired

    return x_cf_sc, rep, meta, C_sc


# ---------------------------------------------------------------------------
# Instrumented benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark_instrumented(
    *,
    X_tr_sc, X_te_sc, y_tr, y_te,
    models: dict,
    factual_indices: np.ndarray,
    methods=("pace", "nice_base", "nice_spars", "mcce", "dice"),
    seed=0,
    guided_n_candidates=2000,
    guided_max_changed_features=8,
    guided_anchor_k=250,
    guided_stageA_random_state=0,
    guided_anchors_random_state=1,
    feature_info=None,
    pre=None,
    X_tr_raw=None,
    X_te_raw=None,
    dice_spec=None,
    dataset_name: str = "",
    nice_base_skip: Optional[set] = None,
    nice_spars_skip: Optional[set] = None,
):
    """
    Same dispatch as tabpfn_cf5.ipynb's run_benchmark, plus per-(model,
    method, factual) target-model call/row instrumentation, plus a
    one-off record of the LR-guidance cost of building PACE's cache.

    Returns (df, cache_build_rows) where df has one row per
    (model, method, idx) with the usual metrics plus `target_calls` /
    `target_rows_evaluated`, and cache_build_rows has one row per
    model_name with the guidance-model cost of guided_cf_build_cache.
    """
    nice_base_skip = nice_base_skip or set()
    nice_spars_skip = nice_spars_skip or set()

    rows = []
    cache_build_rows = []

    weights = DICE_WEIGHT_POLICIES.get(dice_spec["opt_method"], {})

    for model_name, model in models.items():
        print("Working with model", model_name, flush=True)
        pp = proba_fn(model)
        pp_guidance = proba_fn(models["LR"])
        pp_target = proba_fn(model)

        p_te = pp(X_te_sc)
        acc = accuracy_score(y_te, (p_te >= 0.5).astype(int))
        auc = roc_auc_score(y_te, p_te)

        predict_fn_proba = lambda X: model.predict_proba(X)
        _nice_num_cols = dice_spec["continuous_features"] if dice_spec else None
        _nice_cat_cols = dice_spec["categorical_features"] if dice_spec else None
        _X_tr_raw_no_outcome = (
            X_tr_raw.drop(columns=[dice_spec["outcome_name"]], errors="ignore")
            if X_tr_raw is not None and dice_spec else None
        )

        nice_base_obj = nice_base_ctx = None
        if "nice_base" in methods and (model_name, dataset_name) not in nice_base_skip:
            nice_base_obj, nice_base_ctx = build_nice_explainer(
                X_train_sc=X_tr_sc, y_train=y_tr, predict_fn_proba=predict_fn_proba,
                feature_info=feature_info, nice_opt="none",
                X_tr_raw_df=_X_tr_raw_no_outcome, model=model, pre=pre,
                num_cols=_nice_num_cols, cat_cols=_nice_cat_cols,
            )

        nice_spars_obj = nice_spars_ctx = None
        if "nice_spars" in methods and (model_name, dataset_name) not in nice_spars_skip:
            nice_spars_obj, nice_spars_ctx = build_nice_explainer(
                X_train_sc=X_tr_sc, y_train=y_tr, predict_fn_proba=predict_fn_proba,
                feature_info=feature_info, nice_opt="sparsity",
                X_tr_raw_df=_X_tr_raw_no_outcome, model=model, pre=pre,
                num_cols=_nice_num_cols, cat_cols=_nice_cat_cols,
            )

        mcce_obj = mcce_ctx = None
        if "mcce" in methods:
            _mcce_num_cols = dice_spec["continuous_features"] if dice_spec else []
            _mcce_cat_cols = dice_spec["categorical_features"] if dice_spec else []
            _imm_num_names = frozenset(
                _mcce_num_cols[i] for i in getattr(feature_info, "immutable_num", frozenset())
                if i < len(_mcce_num_cols)
            )
            _imm_cat_names = frozenset(
                _mcce_cat_cols[i] for i in getattr(feature_info, "immutable_cat", frozenset())
                if i < len(_mcce_cat_cols)
            )
            _mcce_immutables = (
                ImmutableSpec(num=_imm_num_names, cat=_imm_cat_names)
                if (_imm_num_names or _imm_cat_names) else None
            )
            try:
                mcce_obj, mcce_ctx = build_mcce_explainer(
                    X_tr_sc=X_tr_sc, y_tr=y_tr, model=model, pre=pre,
                    num_cols=_mcce_num_cols, cat_cols=_mcce_cat_cols,
                    seed=seed, immutables=_mcce_immutables,
                )
            except Exception as _e:
                print(f"  MCCE build failed for {model_name}: {_e}", flush=True)

        dice_exp = None
        if "dice" in methods:
            dice_data = dice_ml.Data(
                dataframe=X_tr_raw,
                continuous_features=dice_spec["continuous_features"],
                categorical_features=dice_spec["categorical_features"],
                outcome_name=dice_spec["outcome_name"],
            )
            dice_model = dice_ml.Model(model=dice_spec["models"][model_name], backend="sklearn")
            dice_exp = Dice(dice_data, dice_model, method=dice_spec["opt_method"])
            assert dice_exp.model.model is dice_spec["models"][model_name], (
                f"dice_exp built for wrong model: expected {model_name}"
            )

        guided_cache = None
        if "pace" in methods:
            t0_cache = time.perf_counter()
            with count_predict_proba(models["LR"]) as build_counter:
                guided_cache = guided_cf_build_cache(
                    X_tr_sc=X_tr_sc,
                    predict_proba_target=pp_target,
                    predict_proba_guidance=pp_guidance,
                    stageA_random_state=guided_stageA_random_state,
                    anchor_k=guided_anchor_k,
                    anchors_random_state=guided_anchors_random_state,
                    feature_info=feature_info,
                )
            cache_build_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "guidance_calls": build_counter.n_calls,
                "guidance_rows_evaluated": build_counter.n_rows,
                "build_time_s": float(time.perf_counter() - t0_cache),
            })

        for idx in factual_indices:
            x_f = X_te_sc[idx]
            p_f = float(pp(x_f[None, :])[0])
            yhat_f = int(p_f >= 0.5)
            y_desired = 1 - yhat_f

            for method in methods:
                t0 = time.perf_counter()
                target_calls = 0
                target_rows = 0

                if method == "pace":
                    guided_rs = stable_int_seed(seed, model_name, "guided", int(idx))
                    with count_predict_proba(model) as ctr:
                        x_cf, rep, meta, C = guided_cf_one_repeated(
                            x_f_sc=x_f, cache=guided_cache, predict_proba_target=pp,
                            n_candidates=guided_n_candidates,
                            max_changed_features=guided_max_changed_features,
                            random_state=guided_rs, feature_info=feature_info,
                        )
                    target_calls, target_rows = ctr.n_calls, ctr.n_rows

                elif method == "nice_base":
                    if nice_base_obj is None:
                        continue
                    with count_predict_proba(model) as ctr:
                        out = nice_cf_one(
                            nice_obj=nice_base_obj, x_f_sc=x_f, predict_proba_target=pp,
                            proba_threshold=0.5, tol=1e-6, feature_info=feature_info,
                            repair_onehot=False, nice_ctx=nice_base_ctx,
                        )
                    x_cf, rep, meta = out.x_cf, out.rep, out.meta
                    target_calls, target_rows = ctr.n_calls, ctr.n_rows

                elif method == "nice_spars":
                    if nice_spars_obj is None:
                        continue
                    with count_predict_proba(model) as ctr:
                        out = nice_cf_one(
                            nice_obj=nice_spars_obj, x_f_sc=x_f, predict_proba_target=pp,
                            proba_threshold=0.5, tol=1e-6, feature_info=feature_info,
                            repair_onehot=False, nice_ctx=nice_spars_ctx,
                        )
                    x_cf, rep, meta = out.x_cf, out.rep, out.meta
                    target_calls, target_rows = ctr.n_calls, ctr.n_rows

                elif method == "mcce":
                    if mcce_obj is None:
                        continue
                    with count_predict_proba(model) as ctr:
                        out = mcce_cf_one(
                            mcce_obj=mcce_obj, mcce_ctx=mcce_ctx, x_f_sc=x_f,
                            y_desired=y_desired, model=model, proba_threshold=0.5, k=1000,
                        )
                    x_cf, rep, meta = out.x_cf, out.rep, out.meta
                    target_calls, target_rows = ctr.n_calls, ctr.n_rows

                elif method == "dice":
                    x_f_dice = X_te_raw.drop(columns=[dice_spec["outcome_name"]]).iloc[[idx]]
                    dice_pipeline = dice_spec["models"][model_name]
                    try:
                        with count_predict_proba(dice_pipeline) as ctr:
                            x_cf_dice = dice_exp.generate_counterfactuals(
                                query_instances=x_f_dice, total_CFs=1,
                                desired_class="opposite",
                                features_to_vary=dice_spec["features_to_vary"],
                                permitted_range=dice_spec["permitted_range"],
                                verbose=False, **weights,
                            )
                        target_calls, target_rows = ctr.n_calls, ctr.n_rows
                    except UserConfigValidationException as e:
                        x_cf_dice = None
                        rep = {"ok": False, "status": "dice_invalid_config", "msg": str(e)}
                        meta = {"exception": "UserConfigValidationException", "exception_msg": str(e)}
                        target_calls, target_rows = ctr.n_calls, ctr.n_rows
                    except Exception as e:
                        x_cf_dice = None
                        rep = {"ok": False, "status": "dice_exception", "msg": str(e)}
                        meta = {"exception": type(e).__name__, "exception_msg": str(e)}
                        target_calls, target_rows = ctr.n_calls, ctr.n_rows

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
                            rep["status"] = status
                    else:
                        x_cf_sc = None
                    x_cf = x_cf_sc

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
                else:
                    p_cf = np.nan
                    flip_ok = 0
                    l0, l2, margin = np.nan, np.nan, np.nan

                rows.append({
                    "dataset": dataset_name, "model": model_name, "method": method,
                    "idx": int(idx), "ok": bool(ok), "flip_ok": int(flip_ok),
                    "p_f": float(p_f), "p_cf": float(p_cf) if np.isfinite(p_cf) else np.nan,
                    "margin_cf": float(margin) if np.isfinite(margin) else np.nan,
                    "l0": l0, "l2": l2, "time_s": float(dt),
                    "clf_acc": float(acc), "clf_auc": float(auc),
                    "status": rep.get("status"),
                    "target_calls": int(target_calls),
                    "target_rows_evaluated": int(target_rows),
                })

        print(f"  done model={model_name} ({len(rows)} rows so far)", flush=True)

    return pd.DataFrame(rows), pd.DataFrame(cache_build_rows)
