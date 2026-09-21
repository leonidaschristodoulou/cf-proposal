# ==========================================================================
# E4 pilot: sanity-check the call-counting instrumentation in _e4_lib.py
# before committing to the full representative run.
#
# Fast subset: one small real dataset, LR + RF only (no TabPFN -- this is
# about validating the counter mechanics, not measuring TabPFN cost), a
# handful of factuals, all five methods.
#
# What "passing" looks like:
#  - pace: target_calls == 2 on every successful row (1 singleton p_f call
#    + 1 batched Stage C call), per the verified core.py mechanism.
#  - nice_base/nice_spars/mcce/dice: target_calls > 2, varying per instance
#    (iterative/sequential search), target_rows_evaluated much larger than
#    the candidate-pool sizes PACE uses in a single pass.
# ==========================================================================
from _e4_lib import *

DATASET_NAME = "blood_transfusion"
SEED = 1
N_FACTUAL = 8

if __name__ == "__main__":
    ds_in = load_many([DATASET_NAME], openml_version=2)[0]
    clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=SEED, dice_clip_quantiles=clip_q)

    models = fit_models(X_tr, y_tr, seed=SEED, include_tabpfn=False)
    dice_spec["models"] = fit_models_for_dice(
        X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=SEED,
        models_sc={**models, "TabPFN": models["LR"]},  # dummy so fit_models_for_dice's loop doesn't KeyError
    )
    dice_spec["opt_method"] = choose_dice_method(ds_in.name)

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

    test_models = {"LR": models["LR"], "RF": models["RF"]}

    df, cache_df = run_benchmark_instrumented(
        X_tr_sc=X_tr, X_te_sc=X_te, y_tr=y_tr, y_te=y_te,
        models=test_models,
        factual_indices=factual_indices,
        methods=("pace", "nice_base", "nice_spars", "mcce", "dice"),
        seed=SEED,
        feature_info=feature_info,
        pre=pre,
        X_tr_raw=X_tr_raw,
        X_te_raw=X_te_raw,
        dice_spec=dice_spec,
        dataset_name=ds_in.name,
    )

    pd.set_option("display.width", 160)
    print("\n=== cache build cost (LR guidance) ===")
    print(cache_df)

    print("\n=== per-row summary ===")
    print(df[["model", "method", "idx", "ok", "status", "time_s", "target_calls", "target_rows_evaluated"]]
          .to_string())

    print("\n=== target_calls stats by (model, method) ===")
    print(df.groupby(["model", "method"])["target_calls"].agg(["count", "mean", "min", "max"]))

    print("\n=== target_rows_evaluated stats by (model, method) ===")
    print(df.groupby(["model", "method"])["target_rows_evaluated"].agg(["count", "mean", "min", "max"]))

    # LR-as-target is a real edge case, not a bug: PACE's guidance model is
    # always LR, so when LR is also the *target*, stageB_generate's one
    # guidance-side confidence check (pfn_cf_guided_onehot.py:445) lands on
    # the same object as the target-side calls and the counter can't tell
    # them apart -- giving 3 instead of 2. Confirmed in the full E4 run
    # (e4_tabpfn_cost.joblib): min==max==2 for every non-LR target.
    pace_calls = df.loc[(df["method"] == "pace") & (df["ok"]), ["model", "target_calls"]]
    non_lr = pace_calls.loc[pace_calls["model"] != "LR", "target_calls"]
    lr = pace_calls.loc[pace_calls["model"] == "LR", "target_calls"]
    assert (non_lr == 2).all(), f"Expected PACE target_calls==2 for non-LR targets, got {non_lr.unique()}"
    assert (lr == 3).all(), f"Expected PACE target_calls==3 when target==guidance==LR, got {lr.unique()}"
    print("\nPACE call-count invariant holds: ==2 for non-LR targets, ==3 when target==guidance==LR.")
