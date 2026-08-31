# ==========================================================================
# E5 evidence, part 2: DiCE-genetic vs TabPFN on real-world data -- the
# "third datapoint" for the robustness/decoupling check.
#
# The paper's DiCE-TabPFN collapse is currently only shown under
# genetic-on-synthetic and (post-E5b-fix) random-on-real-world. This adds
# genetic-on-real-world-TabPFN so the conclusion doesn't rest on which
# optimizer was used. Scoped to the 3 real-world datasets with the clearest
# categorical structure (credit-g, australian, sick) rather than all 10, to
# keep compute tractable -- genetic issues far more target-model calls per
# instance than random, and random-vs-TabPFN already showed ~1000x cost
# outliers on some instances (diabetes_binarized). Budgeted generously
# (24h partition max) and checkpointed accordingly.
#
# Writes to a NEW file (dice_genetic_tabpfn_realworld.joblib) -- does not
# touch any existing joblib cache.
# ==========================================================================

from _e5_lib import *
import os

REAL_DATASET_NAMES = ["credit-g", "australian", "sick"]
SEEDS = [1, 3, 5, 7, 9]
N_FACTUAL = 100
CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/dice_genetic_tabpfn_realworld.joblib"

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
            print(f"Resuming: {len(done_pairs)} pair(s) already done.", flush=True)

    print(f"Loading {len(REAL_DATASET_NAMES)} real datasets...", flush=True)
    real_datasets = load_many(REAL_DATASET_NAMES, openml_version=2)
    for ds in real_datasets:
        print(f"  {ds.name}  cat_cols={list(ds.cat_cols)}", flush=True)

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
            dice_spec = force_opt_method(dice_spec, "genetic")

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
                include_vectors=False,
                dataset_name=ds_in.name,
            )

            # Keep only TabPFN rows -- LR was only run to satisfy run_benchmark's
            # internal pp_guidance lookup (already covered by the LR-only sweep).
            df_tp = df[df["model"] == "TabPFN"].copy()
            df_tp["dataset"] = ds_in.name
            df_tp["seed"] = seed
            df_tp["n_cols_model"] = meta.get("n_cols_model")
            df_tp["n_rows"] = meta.get("n_rows")
            df_tp["msg"] = df_tp["rep"].apply(lambda r: r.get("msg") if isinstance(r, dict) else None)

            all_rows.append(df_tp)
            done_pairs.add((ds_in.name, seed))

            df_ckpt = pd.concat(all_rows, ignore_index=True)
            joblib.dump(df_ckpt, CHECKPOINT_PATH)

            dt_pair = time.perf_counter() - t_pair
            n_ok = int(df_tp["ok"].sum())
            print(f"  -> {len(df_tp)} rows, {n_ok} ok ({100*n_ok/len(df_tp):.0f}%), took {dt_pair:.1f}s. "
                  f"Checkpoint saved ({len(done_pairs)}/{n_total_pairs} pairs, {len(df_ckpt)} rows total).",
                  flush=True)

    t_total = time.perf_counter() - t_start
    df_final = pd.concat(all_rows, ignore_index=True)
    print(f"\n===== FINAL COMPLETENESS BY DATASET (TabPFN, genetic) =====", flush=True)
    print(df_final.groupby("dataset")["ok"].agg(["size", "sum", "mean"]).to_string(), flush=True)
    print(f"\n===== STATUS COUNTS =====", flush=True)
    print(df_final["status"].value_counts().to_string(), flush=True)
    print(f"\nDONE. Total wall clock: {t_total/3600:.2f}h", flush=True)
    print(f"Wrote {CHECKPOINT_PATH}", flush=True)
