# ==========================================================================
# E2 evidence: what actually happens when NICE-sparse runs against TabPFN
# on the 5 (dataset) combos currently hardcoded as skipped in run_benchmark
# (_nice_spars_skip: diabetes_binarized, australian, blood_transfusion,
# chronic_kidney_disease, heloc)?
#
# That skip set has no logged reason anywhere -- it's an unexplained
# hardcoded exclusion. This bypasses it (calls build_nice_explainer /
# nice_cf_one directly, not through run_benchmark's skip-checking wrapper)
# to find out empirically: does it crash, hang, time out, or actually work?
# This determines whether "not_run" for these cells is an honest, justified
# label (per E2's schema) or whether NICE-sparse+TabPFN should actually be
# attempted for real on these datasets, the same way DiCE+TabPFN was.
#
# Bounded: 5 factuals per dataset, one dataset at a time, each wrapped in
# its own try/except so a crash/hang on one dataset doesn't lose evidence
# already gathered for the others (flush=True prints after every step so
# partial progress survives even if the job gets killed by the SLURM
# walltime on a hang).
#
# Does not touch any existing joblib cache. Writes a small summary CSV.
# ==========================================================================

from _e5_lib import *
import traceback

SKIPPED_COMBOS = [
    "diabetes_binarized", "australian", "blood_transfusion",
    "chronic_kidney_disease", "heloc",
]
# "diabetes_binarized" isn't a real-world openml name -- it's produced by
# prepare_dataset/load_many from "diabetes". Map back for load_many().
LOAD_NAME_MAP = {"diabetes_binarized": "diabetes"}

SEED = 1        # matches one of the real seeds used in the actual protocol
N_FACTUAL = 5   # small and bounded -- this is a diagnostic, not a full rerun

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    results = []

    for skip_name in SKIPPED_COMBOS:
        load_name = LOAD_NAME_MAP.get(skip_name, skip_name)
        print(f"\n===== {skip_name} (loading as {load_name!r}) =====", flush=True)

        try:
            t0 = time.perf_counter()
            real_datasets = load_many([load_name], openml_version=2)
            ds_in = real_datasets[0]
            print(f"  Loaded {ds_in.name!r} (expected to match {skip_name!r}: "
                  f"{ds_in.name == skip_name}), cat_cols={list(ds_in.cat_cols)}", flush=True)

            clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
            X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
                prepare_dataset(ds_in, rng=SEED, dice_clip_quantiles=clip_q)

            models = fit_models(X_tr, y_tr, seed=SEED, include_tabpfn=True)
            tp_model = models["TabPFN"]
            print(f"  fit_models done ({time.perf_counter()-t0:.1f}s so far)", flush=True)

            predict_fn_proba = lambda X: tp_model.predict_proba(X)

            _num_cols = dice_spec["continuous_features"]
            _cat_cols = dice_spec["categorical_features"]
            _X_tr_raw_no_outcome = X_tr_raw.drop(columns=[dice_spec["outcome_name"]], errors="ignore")

            print("  Building NICE-sparse explainer against TabPFN...", flush=True)
            t_build = time.perf_counter()
            nice_obj, nice_ctx = build_nice_explainer(
                X_train_sc=X_tr,
                y_train=y_tr,
                predict_fn_proba=predict_fn_proba,
                feature_info=feature_info,
                nice_opt='sparsity',
                X_tr_raw_df=_X_tr_raw_no_outcome,
                model=tp_model,
                pre=pre,
                num_cols=_num_cols,
                cat_cols=_cat_cols,
            )
            dt_build = time.perf_counter() - t_build
            print(f"  Explainer built OK in {dt_build:.1f}s", flush=True)

            rng = np.random.default_rng(SEED)
            factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

            for idx in factual_indices:
                x_f = X_te[idx]
                t_inst = time.perf_counter()
                try:
                    out = nice_cf_one(
                        nice_obj=nice_obj,
                        x_f_sc=x_f,
                        predict_proba_target=lambda X: predict_fn_proba(X)[:, 1],
                        proba_threshold=0.5,
                        tol=1e-6,
                        feature_info=feature_info,
                        repair_onehot=False,
                        nice_ctx=nice_ctx,
                    )
                    dt_inst = time.perf_counter() - t_inst
                    status = out.rep.get("status")
                    print(f"    idx={idx} status={status!r} time_s={dt_inst:.2f}", flush=True)
                    results.append(dict(dataset=skip_name, idx=int(idx), stage="generate",
                                         ok=(status == "ok"), status=status, time_s=dt_inst,
                                         build_time_s=dt_build, error=None))
                except Exception as e:
                    dt_inst = time.perf_counter() - t_inst
                    print(f"    idx={idx} EXCEPTION after {dt_inst:.2f}s: {type(e).__name__}: {e}", flush=True)
                    results.append(dict(dataset=skip_name, idx=int(idx), stage="generate",
                                         ok=False, status="exception", time_s=dt_inst,
                                         build_time_s=dt_build, error=f"{type(e).__name__}: {e}"))

        except Exception as e:
            print(f"  BUILD-STAGE EXCEPTION: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append(dict(dataset=skip_name, idx=None, stage="build",
                                 ok=False, status="build_exception", time_s=None,
                                 build_time_s=None, error=f"{type(e).__name__}: {e}"))

    print("\n===== SUMMARY =====", flush=True)
    df = pd.DataFrame(results)
    print(df.to_string(index=False), flush=True)

    out_path = "/nvme/h/lchristodoulou/pace/cf-proposal/nice_spars_tabpfn_skipped_result.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}", flush=True)
