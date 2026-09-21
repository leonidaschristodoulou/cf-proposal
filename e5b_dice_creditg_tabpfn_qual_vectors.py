# ==========================================================================
# Targeted single-instance rerun: DiCE vs TabPFN on credit-g, idx=244, seed=5
# -- the exact factual used in the qualitative example (Fig: real_cf_example_credit_g.pdf).
#
# Purpose: the E5b regeneration (e5b_regen_dice_tabpfn_real.py) never stored raw
# x_f/x_cf/diff vectors (a known, accepted gap -- see REVISION_EVIDENCE_LOG.md),
# so DiCE is missing from the qualitative-example table even though its scalar
# metrics (l0=1, l2=sqrt(2), status=ok) are present in output_realdata.joblib.
# _e5_lib.py's run_benchmark has now been fixed to populate these vectors when
# include_vectors=True (previously accepted but silently unused). This script
# reproduces the EXACT original protocol (same prepare_dataset/fit_models/
# fit_models_for_dice calls, same seed=5, same factual_indices derivation via
# rng.choice) to regenerate just this one factual with vectors captured, rather
# than repeating the full 4870-row real-world sweep.
#
# NOTE ON EXACT REPRODUCIBILITY: DiCE-random's own generate_counterfactuals
# call is not given an explicit random_seed, so its internal sampling draws
# from whatever the global numpy RNG state happens to be -- not independently
# seeded per call. This script cannot guarantee bit-identical reproduction of
# the *specific* candidate DiCE originally returned; it produces a fresh,
# protocol-faithful DiCE-random result for the same factual under the
# corrected pipeline. We check it against the original row's saved scalars
# (l0=1, l2=1.414214, status=ok) as a sanity check that it's landed on an
# equivalent (not necessarily identical) solution.
#
# Writes ONLY this one row to a new file (does not touch output_realdata.joblib).
# ==========================================================================

from _e5_lib import *

DATASET_NAME = "credit-g"
SEED = 5
TARGET_IDX = 244
N_FACTUAL = 100  # must match the original protocol so factual_indices reproduces identically
OUT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/dice_creditg_tabpfn_qual_vectors.joblib"

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    print(f"Loading {DATASET_NAME}...", flush=True)
    ds_in = load_many([DATASET_NAME], openml_version=2)[0]

    clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=SEED, dice_clip_quantiles=clip_q)

    models = fit_models(X_tr, y_tr, seed=SEED, include_tabpfn=True)
    dice_spec['models'] = fit_models_for_dice(
        X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=SEED, models_sc=models
    )
    dice_spec['opt_method'] = choose_dice_method(ds_in.name)
    print("DiCE opt_method:", dice_spec['opt_method'], flush=True)

    rng = np.random.default_rng(SEED)
    factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)
    assert TARGET_IDX in factual_indices, (
        f"idx={TARGET_IDX} not in reproduced factual_indices for seed={SEED} -- "
        f"protocol mismatch, do not proceed."
    )
    print(f"Confirmed idx={TARGET_IDX} is in the reproduced factual_indices set.", flush=True)

    test_models = {"LR": models["LR"], "TabPFN": models["TabPFN"]}

    df, _, _ = run_benchmark(
        X_tr_sc=X_tr, X_te_sc=X_te, y_tr=y_tr, y_te=y_te,
        models=test_models,
        factual_indices=np.array([TARGET_IDX]),
        methods=("dice",),
        seed=SEED,
        feature_info=feature_info,
        pre=pre,
        X_tr_raw=X_tr_raw,
        X_te_raw=X_te_raw,
        dice_spec=dice_spec,
        include_vectors=True,
        dataset_name=ds_in.name,
    )

    df_tp = df[df["model"] == "TabPFN"].copy()
    df_tp["dataset"] = ds_in.name
    df_tp["seed"] = SEED
    df_tp["n_cols_model"] = meta.get("n_cols_model")
    df_tp["n_rows"] = meta.get("n_rows")

    print(df_tp[["dataset", "model", "method", "idx", "seed", "status", "ok", "flip_ok", "l0", "l2"]].to_string())
    print()
    orig = dict(status="ok", ok=True, flip_ok=1.0, l0=1.0, l2=1.414214)
    print(f"Original (vector-less) row for comparison: {orig}")

    joblib.dump(df_tp, OUT_PATH)
    print(f"Wrote {OUT_PATH}", flush=True)
