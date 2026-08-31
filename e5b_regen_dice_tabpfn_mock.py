# ==========================================================================
# E5b full regeneration: corrected DiCE-vs-TabPFN counterfactuals for all
# SYNTHETIC datasets (feeds Fig 2).
#
# Reproduces the EXACT original protocol from tabpfn_cf5.ipynb cell 20
# (generate_make_classification_suite params) and run_over_datasets (seed,
# n_factual=100, factual-index RNG), using the corrected fit_models_for_dice
# / run_benchmark from _e5_lib.py.
#
# Writes ONLY the corrected DiCE+TabPFN rows to a NEW file
# (dice_tabpfn_corrected_mock.joblib) -- does NOT touch output_mockdata.joblib.
# Merging the corrected rows into the master cache is a separate, reviewable
# step. Checkpoints after every dataset so the job can resume if killed.
#
# Timing basis (from bounded feasibility tests on this same GPU/env):
#   mc_ synthetic, DiCE-genetic, TabPFN: ~3.7-4.1s/instance at d=5,
#   ~5.0-5.1s/instance at d=80 (mild scaling with dimensionality)
#   -> ~3000 instances x ~4.5s (blended) ~= 3.75h
# ==========================================================================

from _e5_lib import *
import os

SEED = 2  # the only seed used for mock data in the original benchmark
N_FACTUAL = 100
CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/dice_tabpfn_corrected_mock.joblib"

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    t_start = time.perf_counter()

    all_rows = []
    done_names = set()
    if os.path.exists(CHECKPOINT_PATH):
        df_existing = joblib.load(CHECKPOINT_PATH)
        if isinstance(df_existing, pd.DataFrame) and len(df_existing):
            all_rows.append(df_existing)
            done_names = set(df_existing["dataset"])
            print(f"Resuming from checkpoint: {len(done_names)} dataset(s) already done.", flush=True)

    # Exact params from tabpfn_cf5.ipynb cell 20
    suite = generate_make_classification_suite(
        n_datasets=30,
        n_samples=1000,
        feature_grid=(5, 10, 20, 40, 80),
        redundancy_grid=(0.0, 0.25, 0.5, 0.75),
        informative_frac=0.3,
        class_sep=1.2,
        flip_y=0.1,
        random_state=42,
    )
    print(f"Generated {len(suite)} synthetic datasets.", flush=True)

    n_total = len(suite)
    n_i = 0

    for ds_in in suite:
        n_i += 1
        if ds_in.name in done_names:
            print(f"[{n_i}/{n_total}] Skipping {ds_in.name} (checkpointed)", flush=True)
            continue

        t_pair = time.perf_counter()
        print(f"[{n_i}/{n_total}] {ds_in.name} ...", flush=True)

        X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
            prepare_dataset(ds_in, rng=SEED)

        models = fit_models(X_tr, y_tr, seed=SEED, include_tabpfn=True)
        dice_spec['models'] = fit_models_for_dice(
            X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=SEED, models_sc=models
        )
        dice_spec['opt_method'] = choose_dice_method(ds_in.name)  # "genetic" for all mc_ names
        assert dice_spec['opt_method'] == "genetic"

        rng = np.random.default_rng(SEED)
        factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

        test_models = {"LR": models["LR"], "TabPFN": models["TabPFN"]}

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
            include_vectors=True,
            dataset_name=ds_in.name,
        )

        df_tp = df[df["model"] == "TabPFN"].copy()
        df_tp["dataset"] = ds_in.name
        df_tp["seed"] = SEED
        df_tp["n_cols_model"] = meta.get("n_cols_model")
        df_tp["n_rows"] = meta.get("n_rows")

        all_rows.append(df_tp)
        done_names.add(ds_in.name)

        df_ckpt = pd.concat(all_rows, ignore_index=True)
        joblib.dump(df_ckpt, CHECKPOINT_PATH)

        dt_pair = time.perf_counter() - t_pair
        n_ok = int(df_tp["ok"].sum())
        print(f"  -> {len(df_tp)} rows, {n_ok} ok, took {dt_pair:.1f}s. "
              f"Checkpoint saved ({len(done_names)}/{n_total} datasets, {len(df_ckpt)} rows total).",
              flush=True)

    t_total = time.perf_counter() - t_start
    print(f"\nDONE. Total wall clock: {t_total/3600:.2f}h", flush=True)
    print(f"Wrote {CHECKPOINT_PATH}", flush=True)
