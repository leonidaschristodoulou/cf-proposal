# ==========================================================================
# E2: full-protocol NICE-sparse vs TabPFN for the two combos that turn out
# to be safe to un-skip (diabetes_binarized ~1.2-2.5s/instance, heloc
# ~10-20s/instance -- both comfortably under the agreed 120s common
# timeout; the other 3 originally-skipped combos genuinely hang and stay
# not_run, no new evidence needed there).
#
# Matches the standard real-world protocol: seeds [1,3,5,7,9] x 100
# factuals. Runs each (dataset, seed) pair in its own spawn-context
# subprocess with an overall per-pair wall-clock budget (2400s) as the
# hard backstop -- reliably kills a hang regardless of cause, unlike
# try/except. Writes partial per-instance results incrementally inside
# each subprocess, so even a killed pair leaves real data for whatever
# instances it completed before hanging.
#
# Writes to a NEW file (nice_spars_tabpfn_unskipped.joblib) -- does not
# touch any existing joblib cache. Checkpointed per (dataset, seed) pair.
# ==========================================================================

import multiprocessing as mp
import time
import json
import os

from _e5_lib import *

DATASETS = ["diabetes_binarized", "heloc"]
LOAD_NAME_MAP = {"diabetes_binarized": "diabetes"}
SEEDS = [1, 3, 5, 7, 9]
N_FACTUAL = 100
PER_PAIR_TIMEOUT_S = 2400  # 40 min hard backstop per (dataset, seed) pair
CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/nice_spars_tabpfn_unskipped.joblib"


def _worker(dataset_name: str, load_name: str, seed: int, n_factual: int, out_path: str):
    """Runs in a subprocess. Writes incremental JSON so partial progress
    survives even if this subprocess gets killed from outside."""
    result = {"dataset": dataset_name, "seed": seed, "build_ok": False,
              "build_time_s": None, "instances": [], "build_error": None}

    def _flush():
        with open(out_path, "w") as f:
            json.dump(result, f)

    try:
        real_datasets = load_many([load_name], openml_version=2)
        ds_in = real_datasets[0]

        clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
        X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
            prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)

        models = fit_models(X_tr, y_tr, seed=seed, include_tabpfn=True)
        tp_model = models["TabPFN"]
        predict_fn_proba = lambda X: tp_model.predict_proba(X)

        _num_cols = dice_spec["continuous_features"]
        _cat_cols = dice_spec["categorical_features"]
        _X_tr_raw_no_outcome = X_tr_raw.drop(columns=[dice_spec["outcome_name"]], errors="ignore")

        t_build = time.perf_counter()
        nice_obj, nice_ctx = build_nice_explainer(
            X_train_sc=X_tr, y_train=y_tr, predict_fn_proba=predict_fn_proba,
            feature_info=feature_info, nice_opt='sparsity',
            X_tr_raw_df=_X_tr_raw_no_outcome, model=tp_model, pre=pre,
            num_cols=_num_cols, cat_cols=_cat_cols,
        )
        result["build_ok"] = True
        result["build_time_s"] = time.perf_counter() - t_build
        _flush()

        rng = np.random.default_rng(seed)
        factual_indices = rng.choice(X_te.shape[0], size=min(n_factual, X_te.shape[0]), replace=False)

        for idx in factual_indices:
            x_f = X_te[idx]
            p_f = float(predict_fn_proba(x_f[None, :])[:, 1][0])
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
                result["instances"].append({"idx": int(idx), "status": status, "ok": ok,
                                             "l0": l0, "l2": l2, "time_s": dt, "error": None})
            except Exception as e:
                dt = time.perf_counter() - t_inst
                result["instances"].append({"idx": int(idx), "status": "exception", "ok": False,
                                             "l0": None, "l2": None, "time_s": dt,
                                             "error": f"{type(e).__name__}: {e}"})
            _flush()

    except Exception as e:
        result["build_error"] = f"{type(e).__name__}: {e}"
        _flush()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    all_rows = []

    done_pairs = set()
    if os.path.exists(CHECKPOINT_PATH):
        df_existing = joblib.load(CHECKPOINT_PATH)
        if isinstance(df_existing, pd.DataFrame) and len(df_existing):
            all_rows.append(df_existing)
            done_pairs = set(zip(df_existing["dataset"], df_existing["seed"]))
            print(f"Resuming: {len(done_pairs)} pair(s) already done.", flush=True)

    pairs = [(d, s) for d in DATASETS for s in SEEDS]
    for i, (dataset_name, seed) in enumerate(pairs, 1):
        if (dataset_name, seed) in done_pairs:
            print(f"[{i}/{len(pairs)}] Skipping {dataset_name} seed={seed} (checkpointed)", flush=True)
            continue

        load_name = LOAD_NAME_MAP.get(dataset_name, dataset_name)
        out_path = f"/tmp/nice_spars_unskip_{dataset_name}_{seed}.json"
        if os.path.exists(out_path):
            os.remove(out_path)

        print(f"[{i}/{len(pairs)}] {dataset_name} seed={seed} (timeout={PER_PAIR_TIMEOUT_S}s) ...", flush=True)
        t0 = time.perf_counter()
        p = ctx.Process(target=_worker, args=(dataset_name, load_name, seed, N_FACTUAL, out_path))
        p.start()
        p.join(timeout=PER_PAIR_TIMEOUT_S)

        timed_out = p.is_alive()
        if timed_out:
            print(f"  PAIR TIMEOUT after {PER_PAIR_TIMEOUT_S}s -- killing subprocess", flush=True)
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()

        dt_pair = time.perf_counter() - t0
        partial = {"build_ok": False, "instances": []}
        if os.path.exists(out_path):
            with open(out_path) as f:
                partial.update(json.load(f))

        rows = []
        for inst in partial.get("instances", []):
            rows.append({
                "model": "TabPFN", "method": "nice_spars", "dataset": dataset_name, "seed": seed,
                "idx": inst["idx"], "ok": bool(inst["ok"]), "status": inst["status"],
                "l0": inst["l0"], "l2": inst["l2"], "time_s": inst["time_s"], "error": inst["error"],
            })
        n_completed = len(rows)
        # If this pair was killed mid-flight, the next un-run factual (if any) is
        # marked as a timeout row so the pair's row count still reflects an attempt.
        if timed_out and n_completed < N_FACTUAL:
            rows.append({
                "model": "TabPFN", "method": "nice_spars", "dataset": dataset_name, "seed": seed,
                "idx": None, "ok": False, "status": "timeout",
                "l0": None, "l2": None, "time_s": PER_PAIR_TIMEOUT_S,
                "error": f"pair killed after {n_completed}/{N_FACTUAL} instances completed",
            })

        df_pair = pd.DataFrame(rows)
        all_rows.append(df_pair)
        done_pairs.add((dataset_name, seed))

        df_ckpt = pd.concat(all_rows, ignore_index=True)
        joblib.dump(df_ckpt, CHECKPOINT_PATH)

        n_ok = sum(1 for r in rows if r["ok"])
        print(f"  -> build_ok={partial.get('build_ok')} n_completed={n_completed} n_ok={n_ok} "
              f"timed_out={timed_out} wall_s={dt_pair:.1f}. Checkpoint saved "
              f"({len(done_pairs)}/{len(pairs)} pairs, {len(df_ckpt)} rows total).", flush=True)
        if partial.get("build_error"):
            print(f"  BUILD ERROR: {partial['build_error']}", flush=True)

    df_final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    print("\n===== FINAL COMPLETENESS BY DATASET (NICE-sparse, TabPFN) =====", flush=True)
    print(df_final.groupby("dataset")["ok"].agg(["size", "sum", "mean"]).to_string(), flush=True)
    print("\n===== STATUS COUNTS =====", flush=True)
    print(df_final["status"].value_counts(dropna=False).to_string(), flush=True)
    print(f"\nWrote {CHECKPOINT_PATH}", flush=True)
