# ==========================================================================
# E5 evidence test: does DiCE-genetic fail on real-world (mixed/categorical)
# data?
#
# Purpose (action list item E5): the paper uses DiCE-genetic only on
# synthetic (continuous-only) datasets and DiCE-random only on real-world
# (mixed) datasets, but never compares the two optimizers head-to-head, so
# there's no controlled evidence that genetic was actually the wrong choice
# for mixed data rather than an arbitrary/handicapping split. This script
# forces DiCE-genetic onto a real-world dataset WITH categorical features
# (credit-g: mixed numeric + one-hot categoricals -- blood_transfusion,
# used in the earlier E5b tests, has none, so it can't surface this) and
# logs the actual outcome per instance: success, exception type, and the
# raw exception message (to check for the documented "value outside
# dataset" / invalid one-hot failure modes).
#
# Runs against LR only for this first pass (fast, and DiCE-genetic's
# categorical/one-hot handling is a search-validity issue independent of
# which sklearn-compatible model is wrapped -- if it's broken, it should
# show up against LR already). Does NOT touch output_realdata.joblib or
# the master notebook.
# ==========================================================================

from _e5_lib import *

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    N_FACTUAL = 20   # more instances than the timing tests -- need a failure rate, not just feasibility
    SEED = 0
    DATASET = "credit-g"   # mixed numeric + categorical (checking_status, credit_history, purpose, personal_status, ...)

    t_start = time.perf_counter()

    real_datasets = load_many([DATASET], openml_version=2)
    ds_in = real_datasets[0]
    print(f"Loaded dataset {ds_in.name!r}  num_cols={list(ds_in.num_cols)}  "
          f"cat_cols={list(ds_in.cat_cols)}", flush=True)
    assert len(ds_in.cat_cols) > 0, "need a dataset with categorical features to test genetic's one-hot handling"

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

    # credit-g would normally get "random" (choose_dice_method only picks
    # "genetic" for mc_/synthetic names) -- force "genetic" to characterize
    # its behaviour on mixed data, per E5's spec.
    default_method = choose_dice_method(ds_in.name)
    dice_spec = force_opt_method(dice_spec, "genetic")
    print(f"DiCE opt_method for {ds_in.name!r}: forced to {dice_spec['opt_method']!r} "
          f"(default would have been {default_method!r})", flush=True)

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)
    print(f"Factual indices: {factual_indices.tolist()}", flush=True)

    test_models = {"LR": models["LR"]}

    print("Running DiCE-genetic against LR on real-world mixed data...", flush=True)
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

    df["msg"] = df["rep"].apply(lambda r: r.get("msg") if isinstance(r, dict) else None)

    print("\n===== PER-INSTANCE RESULTS =====", flush=True)
    print(df[["model", "idx", "status", "ok", "l0", "l2", "time_s", "msg"]].to_string(index=False), flush=True)

    print("\n===== STATUS COUNTS (this is the E5 completeness evidence) =====", flush=True)
    print(df["status"].value_counts(dropna=False).to_string(), flush=True)

    print("\n===== SAMPLE ERROR MESSAGES BY STATUS =====", flush=True)
    for status, grp in df[df["msg"].notna()].groupby("status"):
        print(f"-- {status} ({len(grp)} instances) --", flush=True)
        for msg in grp["msg"].drop_duplicates().head(3):
            print(f"   {msg[:300]}", flush=True)

    print(f"\nTotal wall clock for this script: {t_total:.1f}s", flush=True)

    out_path = "/nvme/h/lchristodoulou/pace/cf-proposal/e5_test3_result.csv"
    df.drop(columns=["rep", "changed_idx"], errors="ignore").to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)
