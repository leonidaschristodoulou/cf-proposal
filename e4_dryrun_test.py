# Quick CPU-only dry run of the E4 driver's logic (LR+RF only, tiny factual
# count) on the synthetic set and one real mixed set, to catch dataset-
# specific bugs (categorical handling, DiCE spec assembly, MCCE build)
# before spending GPU allocation on the full TabPFN sbatch run.
from _e4_lib import *
from e4_tabpfn_cost_instrumentation import DATASET_LOADERS, METHODS, NICE_BASE_SKIP, NICE_SPARS_SKIP

SEED = 1
N_FACTUAL = 4

for dataset_name in ["credit-g"]:
    print(f"\n########## {dataset_name} ##########", flush=True)
    ds_in = DATASET_LOADERS[dataset_name]()
    clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=SEED, dice_clip_quantiles=clip_q)

    models = fit_models(X_tr, y_tr, seed=SEED, include_tabpfn=False)
    dice_spec["models"] = fit_models_for_dice(
        X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=SEED,
        models_sc={**models, "TabPFN": models["LR"]},
    )
    dice_spec["opt_method"] = choose_dice_method(ds_in.name)
    print("dice opt_method:", dice_spec["opt_method"], flush=True)

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

    test_models = {"LR": models["LR"], "RF": models["RF"]}

    df, cache_df = run_benchmark_instrumented(
        X_tr_sc=X_tr, X_te_sc=X_te, y_tr=y_tr, y_te=y_te,
        models=test_models,
        factual_indices=factual_indices,
        methods=METHODS,
        seed=SEED,
        feature_info=feature_info,
        pre=pre,
        X_tr_raw=X_tr_raw,
        X_te_raw=X_te_raw,
        dice_spec=dice_spec,
        dataset_name=ds_in.name,
        nice_base_skip=NICE_BASE_SKIP,
        nice_spars_skip=NICE_SPARS_SKIP,
    )
    print(df[["model", "method", "idx", "ok", "status", "time_s", "target_calls", "target_rows_evaluated"]]
          .to_string(), flush=True)
    print(cache_df, flush=True)

print("\nDRY RUN OK", flush=True)
