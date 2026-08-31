# ==========================================================================
# E2 evidence, continued: NICE-sparse vs TabPFN on the remaining 3 skipped
# combos (blood_transfusion, chronic_kidney_disease, heloc).
#
# v1 (e2_test_nice_spars_tabpfn_skipped.py) found diabetes_binarized works
# fine but hung indefinitely on australian, burning the whole 3h job before
# ever reaching these 3. A plain try/except can't catch a hang -- only a
# hard kill can. This version runs each dataset's diagnostic in its own
# subprocess (spawn context, safe with CUDA) with a bounded per-dataset
# timeout; if it doesn't finish in time, the subprocess is killed and the
# combo is recorded as "hang" rather than eating the rest of the job.
#
# Does not touch any existing joblib cache. Writes a small summary CSV.
# ==========================================================================

import multiprocessing as mp
import time
import json
import os

# Module-level import (not inside _worker) because "from module import *" is
# only legal at module level in Python. Under the "spawn" multiprocessing
# context, each subprocess re-executes this module from scratch anyway, so
# this correctly re-imports everything fresh in every child process.
from _e5_lib import *
import traceback

REMAINING_COMBOS = ["blood_transfusion", "chronic_kidney_disease", "heloc"]
SEED = 1
N_FACTUAL = 5
PER_DATASET_TIMEOUT_S = 300  # 5 min -- v1's hang was evident within seconds of the first call


def _worker(dataset_name: str, seed: int, n_factual: int, out_path: str):
    """Runs in a subprocess (spawned fresh, module reimported). Writes a JSON
    result file; killed from outside on timeout."""
    result = {"dataset": dataset_name, "build_ok": False, "build_time_s": None,
              "instances": [], "build_error": None}

    try:
        real_datasets = load_many([dataset_name], openml_version=2)
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

        rng = np.random.default_rng(seed)
        factual_indices = rng.choice(X_te.shape[0], size=min(n_factual, X_te.shape[0]), replace=False)

        for idx in factual_indices:
            x_f = X_te[idx]
            t_inst = time.perf_counter()
            try:
                out = nice_cf_one(
                    nice_obj=nice_obj, x_f_sc=x_f,
                    predict_proba_target=lambda X: predict_fn_proba(X)[:, 1],
                    proba_threshold=0.5, tol=1e-6, feature_info=feature_info,
                    repair_onehot=False, nice_ctx=nice_ctx,
                )
                dt = time.perf_counter() - t_inst
                result["instances"].append({"idx": int(idx), "status": out.rep.get("status"),
                                             "time_s": dt, "error": None})
            except Exception as e:
                dt = time.perf_counter() - t_inst
                result["instances"].append({"idx": int(idx), "status": "exception",
                                             "time_s": dt, "error": f"{type(e).__name__}: {e}"})
            # Flush partial progress after every instance, in case we get killed mid-dataset
            with open(out_path, "w") as f:
                json.dump(result, f)

    except Exception as e:
        result["build_error"] = f"{type(e).__name__}: {e}"
        result["build_traceback"] = traceback.format_exc()

    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")  # required for safety with CUDA
    all_results = []

    for dataset_name in REMAINING_COMBOS:
        out_path = f"/tmp/nice_spars_v2_{dataset_name}.json"
        if os.path.exists(out_path):
            os.remove(out_path)

        print(f"\n===== {dataset_name} (timeout={PER_DATASET_TIMEOUT_S}s) =====", flush=True)
        t0 = time.perf_counter()
        p = ctx.Process(target=_worker, args=(dataset_name, SEED, N_FACTUAL, out_path))
        p.start()
        p.join(timeout=PER_DATASET_TIMEOUT_S)

        timed_out = p.is_alive()
        if timed_out:
            print(f"  TIMEOUT after {PER_DATASET_TIMEOUT_S}s -- killing subprocess", flush=True)
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
                p.join()

        dt = time.perf_counter() - t0
        partial = {"dataset": dataset_name, "timed_out": timed_out, "wall_s": dt}
        if os.path.exists(out_path):
            with open(out_path) as f:
                partial.update(json.load(f))
        else:
            partial["note"] = "no output file -- killed before writing anything"

        print(f"  build_ok={partial.get('build_ok')} "
              f"n_instances_completed={len(partial.get('instances', []))} "
              f"timed_out={timed_out} wall_s={dt:.1f}", flush=True)
        for inst in partial.get("instances", []):
            print(f"    idx={inst['idx']} status={inst['status']!r} time_s={inst['time_s']:.2f} "
                  f"error={inst['error']}", flush=True)
        if partial.get("build_error"):
            print(f"  BUILD ERROR: {partial['build_error']}", flush=True)

        all_results.append(partial)

    print("\n===== SUMMARY =====", flush=True)
    import pandas as pd
    rows = []
    for r in all_results:
        if r.get("instances"):
            for inst in r["instances"]:
                rows.append({"dataset": r["dataset"], "idx": inst["idx"], "status": inst["status"],
                             "time_s": inst["time_s"], "error": inst["error"],
                             "timed_out": r["timed_out"]})
        else:
            rows.append({"dataset": r["dataset"], "idx": None,
                         "status": "timeout" if r["timed_out"] else "build_error",
                         "time_s": r.get("wall_s"), "error": r.get("build_error"),
                         "timed_out": r["timed_out"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False), flush=True)

    out_csv = "/nvme/h/lchristodoulou/pace/cf-proposal/nice_spars_tabpfn_skipped_result_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}", flush=True)
