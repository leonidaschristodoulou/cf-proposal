# ==========================================================================
# E5b full regeneration: corrected DiCE-vs-TabPFN counterfactuals for all
# REAL-WORLD datasets (feeds Fig 5).
#
# Reproduces the EXACT original protocol from tabpfn_cf5.ipynb cells 19-20
# (dataset list) and run_over_datasets (seeds, n_factual=100, factual-index
# RNG), using the corrected fit_models_for_dice / run_benchmark from
# _e5_lib.py (TabPFN added to the DiCE clone loop; dice_exp always rebuilt
# per model_name instead of silently reusing RF's stale explainer).
#
# Writes ONLY the corrected DiCE+TabPFN rows to a NEW file
# (dice_tabpfn_corrected_real.joblib) -- does NOT touch output_realdata.joblib.
# Merging the corrected rows into the master cache is a separate, reviewable
# step. Checkpoints after every (dataset, seed) pair so the job can resume
# if it needs to be killed/resubmitted.
#
# Timing basis (from bounded feasibility tests on this same GPU/env):
#   blood_transfusion, DiCE-random, TabPFN: ~1.9s/instance
#   -> ~4870 instances x ~1.9s ~= 2.6h, plus per-pair setup (~2.6s x 50 ~= negligible)
# ==========================================================================

from _e5_lib import *
import os

REAL_DATASET_NAMES = [
    "credit-g", "breast_cancer", "diabetes", "ilpd",
    "australian", "heart_disease",
    "sick", "heloc", "blood_transfusion", "chronic_kidney_disease",
]
SEEDS = [1, 3, 5, 7, 9]
N_FACTUAL = 100
CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/dice_tabpfn_corrected_real.joblib"

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    t_start = time.perf_counter()

    all_rows = []
    done_pairs = set()
    if os.path.exists(CHECKPOINT_PATH):
        df_existing = joblib.load(CHECKPOINT_PATH)
        if isinstance(df_existing, pd.DataFrame) and len(df_existing):
            all_rows.append(df_existing)
            done_pairs = set(zip(df_existing["dataset"], df_existing["seed"]))
            print(f"Resuming from checkpoint: {len(done_pairs)} (dataset, seed) pair(s) already done.", flush=True)

    print(f"Loading {len(REAL_DATASET_NAMES)} real datasets...", flush=True)
    real_datasets = load_many(REAL_DATASET_NAMES, openml_version=2)
    for ds in real_datasets:
        print(f"  {ds.name}", flush=True)

    n_total_pairs = len(real_datasets) * len(SEEDS)
    n_pair = 0

    for ds_in in real_datasets:
        for seed in SEEDS:
            n_pair += 1
            if (ds_in.name, seed) in done_pairs:
                print(f"[{n_pair}/{n_total_pairs}] Skipping {ds_in.name} seed={seed} (checkpointed)", flush=True)
                continue

            t_pair = time.perf_counter()
            print(f"[{n_pair}/{n_total_pairs}] {ds_in.name} seed={seed} ...", flush=True)

            clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
            X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
                prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)

            models = fit_models(X_tr, y_tr, seed=seed, include_tabpfn=True)
            dice_spec['models'] = fit_models_for_dice(
                X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=seed, models_sc=models
            )
            dice_spec['opt_method'] = choose_dice_method(ds_in.name)  # "random" for all real-world names

            rng = np.random.default_rng(seed)
            factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

            test_models = {"LR": models["LR"], "TabPFN": models["TabPFN"]}

            df, _, _ = run_benchmark(
                X_tr_sc=X_tr, X_te_sc=X_te, y_tr=y_tr, y_te=y_te,
                models=test_models,
                factual_indices=factual_indices,
                methods=("dice",),
                seed=seed,
                feature_info=feature_info,
                pre=pre,
                X_tr_raw=X_tr_raw,
                X_te_raw=X_te_raw,
                dice_spec=dice_spec,
                include_vectors=True,
                dataset_name=ds_in.name,
            )

            # Keep only the corrected TabPFN rows -- LR was run only to satisfy
            # run_benchmark's internal pp_guidance lookup and was never buggy.
            df_tp = df[df["model"] == "TabPFN"].copy()
            df_tp["dataset"] = ds_in.name
            df_tp["seed"] = seed
            df_tp["n_cols_model"] = meta.get("n_cols_model")
            df_tp["n_rows"] = meta.get("n_rows")

            all_rows.append(df_tp)
            done_pairs.add((ds_in.name, seed))

            df_ckpt = pd.concat(all_rows, ignore_index=True)
            joblib.dump(df_ckpt, CHECKPOINT_PATH)

            dt_pair = time.perf_counter() - t_pair
            n_ok = int(df_tp["ok"].sum())
            print(f"  -> {len(df_tp)} rows, {n_ok} ok, took {dt_pair:.1f}s. "
                  f"Checkpoint saved ({len(done_pairs)}/{n_total_pairs} pairs, {len(df_ckpt)} rows total).",
                  flush=True)

    t_total = time.perf_counter() - t_start
    print(f"\nDONE. Total wall clock: {t_total/3600:.2f}h", flush=True)
    print(f"Wrote {CHECKPOINT_PATH}", flush=True)
