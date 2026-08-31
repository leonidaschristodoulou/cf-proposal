# ==========================================================================
# E5b dimensionality-scaling check: DiCE-genetic vs TabPFN on an 80-feature
# synthetic dataset (the largest in the mock suite).
#
# Test 2 (e5_test2_...) only checked a 5-feature synthetic dataset and found
# ~3.7-4.1s/instance. Genetic search cost plausibly scales with feature
# count, and the mock suite goes up to 80 features, so this checks the
# worst-case per-instance cost before committing to a full regeneration
# scope/time-budget for output_mockdata.joblib (30 datasets, up to f80).
# ==========================================================================

from _e5_lib import *

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    N_FACTUAL = 2   # tightly bounded -- just need a per-instance time estimate at f80
    SEED = 0

    t_start = time.perf_counter()

    suite = generate_make_classification_suite(
        n_datasets=1,
        n_samples=1000,
        feature_grid=(80,),
        redundancy_grid=(0.25,),
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
    print(f"DiCE opt_method for {ds_in.name!r}: {dice_spec['opt_method']!r}", flush=True)
    assert dice_spec['opt_method'] == "genetic"

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)
    print(f"Factual indices: {factual_indices.tolist()}", flush=True)

    test_models = {"LR": models["LR"], "TabPFN": models["TabPFN"]}

    print("Running DiCE-genetic against LR and TabPFN at d=80...", flush=True)
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
    print(f"\nTotal wall clock for this script: {t_total:.1f}s", flush=True)
