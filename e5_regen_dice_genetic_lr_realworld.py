# ==========================================================================
# E5 evidence, part 1: DiCE-genetic completeness across ALL real-world
# datasets, against LR.
#
# Test 3 (e5_test3_...) only characterized credit-g. This extends the same
# idea (force opt_method="genetic" instead of the default "random") across
# all 10 real-world datasets, matching the original protocol (5 seeds, 100
# factuals). LR only: genetic's one-hot/categorical handling is a
# search-validity property of DiCE itself, not the target classifier, so LR
# is sufficient evidence for "was genetic a clean option on mixed data" --
# no need to repeat against RF/XGB/TabPFN for this specific question.
#
# Writes to a NEW file (dice_genetic_lr_realworld.joblib) -- does not touch
# any existing joblib cache. Checkpoints per (dataset, seed) pair.
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
CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/dice_genetic_lr_realworld.joblib"

if __name__ == "__main__":
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
            default_method = choose_dice_method(ds_in.name)
            dice_spec = force_opt_method(dice_spec, "genetic")

            rng = np.random.default_rng(seed)
            factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

            # LR only -- genetic's categorical-handling issue is optimizer-level, not model-level
            test_models = {"LR": models["LR"]}

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

            df["dataset"] = ds_in.name
            df["seed"] = seed
            df["n_cols_model"] = meta.get("n_cols_model")
            df["n_rows"] = meta.get("n_rows")
            df["msg"] = df["rep"].apply(lambda r: r.get("msg") if isinstance(r, dict) else None)

            all_rows.append(df)
            done_pairs.add((ds_in.name, seed))

            df_ckpt = pd.concat(all_rows, ignore_index=True)
            joblib.dump(df_ckpt, CHECKPOINT_PATH)

            dt_pair = time.perf_counter() - t_pair
            n_ok = int(df["ok"].sum())
            print(f"  -> {len(df)} rows, {n_ok} ok ({100*n_ok/len(df):.0f}%), took {dt_pair:.1f}s. "
                  f"Checkpoint saved ({len(done_pairs)}/{n_total_pairs} pairs, {len(df_ckpt)} rows total).",
                  flush=True)

    t_total = time.perf_counter() - t_start
    df_final = pd.concat(all_rows, ignore_index=True)
    print(f"\n===== FINAL COMPLETENESS BY DATASET =====", flush=True)
    print(df_final.groupby("dataset")["ok"].agg(["size", "sum", "mean"]).to_string(), flush=True)
    print(f"\n===== STATUS COUNTS =====", flush=True)
    print(df_final["status"].value_counts().to_string(), flush=True)
    print(f"\nDONE. Total wall clock: {t_total/3600:.2f}h", flush=True)
    print(f"Wrote {CHECKPOINT_PATH}", flush=True)
