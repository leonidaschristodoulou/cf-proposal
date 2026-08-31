# ==========================================================================
# E5b feasibility test #2: DiCE-genetic vs TabPFN on a synthetic dataset.
#
# Purpose: the first E5b test (e5b_test_dice_tabpfn.py) showed DiCE-"random"
# against TabPFN on real-world data is fast (~1.9s/instance). But synthetic
# datasets (Fig 2) use DiCE's "genetic" optimizer instead (population-based
# evolutionary search), which is architecturally the expensive case -- each
# generation issues fresh target-model calls. This test checks whether a
# real (non-stale) DiCE-genetic explainer against TabPFN is tractable on a
# small synthetic dataset, before committing to a full E5b rerun for Fig 2.
#
# Uses the same corrected fit_models_for_dice / run_benchmark from _e5_lib.py
# (TabPFN added to the DiCE pipeline clone loop; dice_exp always rebuilt per
# model_name). Does NOT touch output_mockdata.joblib or the master notebook.
# ==========================================================================

from _e5_lib import *

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    N_FACTUAL = 3   # kept small -- genetic cost per instance is the unknown here
    SEED = 0

    t_start = time.perf_counter()

    # One small synthetic dataset: 5 features, no redundancy, 1000 samples.
    # generate_make_classification_suite names datasets "mc_...", so
    # choose_dice_method() will select "genetic" automatically, matching
    # exactly how Fig 2's synthetic datasets are run in the real benchmark.
    suite = generate_make_classification_suite(
        n_datasets=1,
        n_samples=1000,
        feature_grid=(5,),
        redundancy_grid=(0.0,),
        random_state=42,
    )
    ds_in = suite[0]
    print(f"Loaded synthetic dataset {ds_in.name!r}", flush=True)

    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=SEED)

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
    print(f"DiCE opt_method for {ds_in.name!r}: {dice_spec['opt_method']!r} "
          f"(expect 'genetic')", flush=True)
    assert dice_spec['opt_method'] == "genetic", "expected genetic for a synthetic dataset"

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)
    print(f"Factual indices: {factual_indices.tolist()}", flush=True)

    test_models = {"LR": models["LR"], "TabPFN": models["TabPFN"]}

    print("Running DiCE-genetic against LR and TabPFN...", flush=True)
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

    out_path = "/nvme/h/lchristodoulou/pace/cf-proposal/e5_test2_result.csv"
    df.drop(columns=["rep", "changed_idx"], errors="ignore").to_csv(out_path, index=False)
    print(f"Wrote {out_path}", flush=True)
