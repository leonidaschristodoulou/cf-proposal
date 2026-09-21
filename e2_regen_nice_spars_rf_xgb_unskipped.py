# ==========================================================================
# E2: characterize the 6 NICE-sparse skip entries we hadn't looked at yet --
# RF on australian/diabetes_binarized/heloc, XGB on australian/blood_transfusion/heloc.
# The 5 TabPFN skips were already resolved (3 genuine hangs, 2 safe-to-run);
# this covers the remaining entries in _nice_spars_skip.
#
# Same design as e2_regen_nice_spars_tabpfn_unskipped_v2.py: a persistent
# worker subprocess builds the explainer once per (model, dataset, seed) and
# processes factuals one at a time over a Queue; if the parent gets no
# response within the agreed 120s common timeout, it kills only that worker
# (losing the one stuck instance, marked "timeout"), spawns a fresh one, and
# continues. RF/XGB inference is cheap (no GPU needed), so this goes straight
# for the full standard protocol rather than a small diagnostic first --
# the timeout mechanism bounds the cost regardless of scale.
#
# Writes to a NEW file (nice_spars_rf_xgb_unskipped.joblib) -- does not touch
# any existing joblib cache. Checkpointed per (model, dataset, seed) pair.
# ==========================================================================

import multiprocessing as mp
import queue
import time
import os

from _e5_lib import *

PAIRS = [
    ("RF", "australian"),
    ("RF", "diabetes_binarized"),
    ("RF", "heloc"),
    ("XGB", "australian"),
    ("XGB", "blood_transfusion"),
    ("XGB", "heloc"),
]
LOAD_NAME_MAP = {"diabetes_binarized": "diabetes"}
SEEDS = [1, 3, 5, 7, 9]
N_FACTUAL = 100
PER_INSTANCE_TIMEOUT_S = 120  # the agreed common rule
BUILD_TIMEOUT_S = 120
CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/nice_spars_rf_xgb_unskipped.joblib"


def _worker(model_name: str, dataset_name: str, load_name: str, seed: int, task_q, result_q):
    """Persistent worker: builds the explainer once, then processes one
    factual index at a time from task_q, writing results to result_q."""
    try:
        real_datasets = load_many([load_name], openml_version=2)
        ds_in = real_datasets[0]

        clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
        X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
            prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)

        models = fit_models(X_tr, y_tr, seed=seed, include_tabpfn=False)
        target_model = models[model_name]
        predict_fn_proba = lambda X: target_model.predict_proba(X)

        _num_cols = dice_spec["continuous_features"]
        _cat_cols = dice_spec["categorical_features"]
        _X_tr_raw_no_outcome = X_tr_raw.drop(columns=[dice_spec["outcome_name"]], errors="ignore")

        t_build = time.perf_counter()
        nice_obj, nice_ctx = build_nice_explainer(
            X_train_sc=X_tr, y_train=y_tr, predict_fn_proba=predict_fn_proba,
            feature_info=feature_info, nice_opt='sparsity',
            X_tr_raw_df=_X_tr_raw_no_outcome, model=target_model, pre=pre,
            num_cols=_num_cols, cat_cols=_cat_cols,
        )
        result_q.put(("__ready__", time.perf_counter() - t_build))
    except Exception as e:
        result_q.put(("__build_error__", f"{type(e).__name__}: {e}"))
        return

    while True:
        idx = task_q.get()
        if idx is None:
            return
        x_f = X_te[int(idx)]
        t_inst = time.perf_counter()
        try:
            out = nice_cf_one(
                nice_obj=nice_obj, x_f_sc=x_f,
                predict_proba_target=lambda X: predict_fn_proba(X)[:, 1],
                proba_threshold=0.5, tol=1e-6, feature_info=feature_info,
                repair_onehot=False, nice_ctx=nice_ctx,
            )
            dt = time.perf_counter() - t_inst
            status = out.rep.get("status")
            ok = (status == "ok") and (out.x_cf is not None)
            l0 = l2 = None
            if ok:
                diff = out.x_cf - x_f
                l0 = l0_mixed(x_f, out.x_cf, feature_info, tol=1e-6)
                l2 = float(np.sqrt(np.sum(diff * diff)))
            result_q.put(("__result__", int(idx), status, ok, l0, l2, dt, None))
        except Exception as e:
            dt = time.perf_counter() - t_inst
            result_q.put(("__result__", int(idx), "exception", False, None, None, dt,
                           f"{type(e).__name__}: {e}"))


def _spawn_worker(ctx, model_name, dataset_name, load_name, seed):
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(model_name, dataset_name, load_name, seed, task_q, result_q))
    p.start()
    return p, task_q, result_q


def run_pair(ctx, model_name, dataset_name, load_name, seed, factual_indices):
    rows = []
    p, task_q, result_q = _spawn_worker(ctx, model_name, dataset_name, load_name, seed)

    try:
        msg = result_q.get(timeout=BUILD_TIMEOUT_S)
    except queue.Empty:
        p.terminate(); p.join(timeout=10)
        if p.is_alive():
            p.kill(); p.join()
        print(f"    BUILD TIMEOUT after {BUILD_TIMEOUT_S}s -- entire pair marked timeout", flush=True)
        return [{"model": model_name, "method": "nice_spars", "dataset": dataset_name, "seed": seed,
                  "idx": None, "ok": False, "status": "build_timeout", "l0": None, "l2": None,
                  "time_s": BUILD_TIMEOUT_S, "error": "explainer construction did not return"}]

    if msg[0] == "__build_error__":
        p.join(timeout=5)
        print(f"    BUILD ERROR: {msg[1]}", flush=True)
        return [{"model": model_name, "method": "nice_spars", "dataset": dataset_name, "seed": seed,
                  "idx": None, "ok": False, "status": "build_exception", "l0": None, "l2": None,
                  "time_s": None, "error": msg[1]}]

    build_time_s = msg[1]
    print(f"    explainer built in {build_time_s:.2f}s", flush=True)

    remaining = list(factual_indices)
    n_stalls = 0
    while remaining:
        idx = remaining[0]
        task_q.put(int(idx))
        try:
            msg = result_q.get(timeout=PER_INSTANCE_TIMEOUT_S)
            _, ridx, status, ok, l0, l2, dt, err = msg
            rows.append({"model": model_name, "method": "nice_spars", "dataset": dataset_name,
                         "seed": seed, "idx": ridx, "ok": ok, "status": status,
                         "l0": l0, "l2": l2, "time_s": dt, "error": err})
            remaining.pop(0)
        except queue.Empty:
            n_stalls += 1
            p.terminate(); p.join(timeout=10)
            if p.is_alive():
                p.kill(); p.join()
            rows.append({"model": model_name, "method": "nice_spars", "dataset": dataset_name,
                         "seed": seed, "idx": int(idx), "ok": False, "status": "timeout",
                         "l0": None, "l2": None, "time_s": PER_INSTANCE_TIMEOUT_S,
                         "error": f"no response within {PER_INSTANCE_TIMEOUT_S}s"})
            remaining.pop(0)
            if remaining:
                p, task_q, result_q = _spawn_worker(ctx, model_name, dataset_name, load_name, seed)
                try:
                    msg = result_q.get(timeout=BUILD_TIMEOUT_S)
                    if msg[0] == "__build_error__":
                        print(f"    rebuild failed: {msg[1]} -- aborting remainder of pair", flush=True)
                        for ridx in remaining:
                            rows.append({"model": model_name, "method": "nice_spars", "dataset": dataset_name,
                                         "seed": seed, "idx": int(ridx), "ok": False, "status": "build_exception",
                                         "l0": None, "l2": None, "time_s": None, "error": msg[1]})
                        remaining = []
                except queue.Empty:
                    print(f"    rebuild timed out -- aborting remainder of pair", flush=True)
                    for ridx in remaining:
                        rows.append({"model": model_name, "method": "nice_spars", "dataset": dataset_name,
                                     "seed": seed, "idx": int(ridx), "ok": False, "status": "build_timeout",
                                     "l0": None, "l2": None, "time_s": BUILD_TIMEOUT_S,
                                     "error": "explainer rebuild did not return"})
                    remaining = []

    try:
        task_q.put(None)
        p.join(timeout=5)
        if p.is_alive():
            p.terminate(); p.join(timeout=5)
            if p.is_alive():
                p.kill(); p.join()
    except Exception:
        pass

    print(f"    {len(rows)}/{len(factual_indices)} instances processed, {n_stalls} stall(s)", flush=True)
    return rows


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    all_rows = []

    done_pairs = set()
    if os.path.exists(CHECKPOINT_PATH):
        df_existing = joblib.load(CHECKPOINT_PATH)
        if isinstance(df_existing, pd.DataFrame) and len(df_existing):
            all_rows.append(df_existing)
            done_pairs = set(zip(df_existing["model"], df_existing["dataset"], df_existing["seed"]))
            print(f"Resuming: {len(done_pairs)} pair(s) already done.", flush=True)

    triples = [(m, d, s) for (m, d) in PAIRS for s in SEEDS]
    for i, (model_name, dataset_name, seed) in enumerate(triples, 1):
        if (model_name, dataset_name, seed) in done_pairs:
            print(f"[{i}/{len(triples)}] Skipping {model_name}/{dataset_name} seed={seed} (checkpointed)", flush=True)
            continue

        print(f"[{i}/{len(triples)}] {model_name}/{dataset_name} seed={seed} ...", flush=True)
        t0 = time.perf_counter()

        load_name = LOAD_NAME_MAP.get(dataset_name, dataset_name)

        real_datasets = load_many([load_name], openml_version=2)
        ds_in = real_datasets[0]
        clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
        _, X_te, _, _, _, _, _, _, _, _ = prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)
        rng = np.random.default_rng(seed)
        factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

        rows = run_pair(ctx, model_name, dataset_name, load_name, seed, factual_indices)
        df_pair = pd.DataFrame(rows)
        all_rows.append(df_pair)
        done_pairs.add((model_name, dataset_name, seed))

        df_ckpt = pd.concat(all_rows, ignore_index=True)
        joblib.dump(df_ckpt, CHECKPOINT_PATH)

        n_ok = sum(1 for r in rows if r["ok"])
        dt_pair = time.perf_counter() - t0
        print(f"  -> {len(rows)} rows, {n_ok} ok, took {dt_pair:.1f}s. "
              f"Checkpoint saved ({len(done_pairs)}/{len(triples)} pairs, {len(df_ckpt)} rows total).",
              flush=True)

    df_final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    print("\n===== FINAL COMPLETENESS BY (model, dataset) =====", flush=True)
    print(df_final.groupby(["model", "dataset"])["ok"].agg(["size", "sum", "mean"]).to_string(), flush=True)
    print("\n===== STATUS COUNTS =====", flush=True)
    print(df_final["status"].value_counts(dropna=False).to_string(), flush=True)
    print(f"\nWrote {CHECKPOINT_PATH}", flush=True)
