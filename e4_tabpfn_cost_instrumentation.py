# ==========================================================================
# E4: instrument target-model call counts / row-evaluations for all five
# methods (pace, nice_base, nice_spars, mcce, dice) across a representative
# dataset x classifier grid, to demonstrate (not just assert) that PACE's
# efficiency comes from a single batched target-model pass while baselines
# query the target iteratively.
#
# This run also doubles as E1b's empirical anchor: DiCE-genetic runs on the
# synthetic (continuous-only) dataset here (choose_dice_method picks
# "genetic" for names starting "mc_"), giving a real measured call-count
# for "GA-style search is call-hungry" without needing a MOC integration.
#
# Dataset choice (4, spanning continuous-only / mixed / the hard TabPFN
# case, same spirit as A1's representative subset):
#   - mc_f80_red40_inf24_seed787359109: synthetic, 80 continuous features,
#     identical instance used by A1 (see a1_extra_datasets_ablation.py) --
#     reused here so this dataset's DiCE run uses "genetic".
#   - breast_cancer: mixed, the hard TabPFN-incompleteness case flagged
#     throughout the revision.
#   - credit-g: mixed, moderate size, well-characterized elsewhere.
#   - heloc: mixed, larger n_rows, real TabPFN-scaling case (NICE-sparse
#     already confirmed ~90% completeness here under the 120s cap in E2).
#
# Small factual count (this is a mechanism-demonstration/instrumentation
# study -- PACE's call count is ~deterministic; the point is to have real,
# runtime-paired call counts to plot, not to re-establish completeness
# statistics that already exist elsewhere) -- 15 factuals x 1 seed x 4
# datasets x 4 classifiers x 5 methods.
#
# nice_base/nice_spars skip sets reused from tabpfn_cf5.ipynb's run_benchmark
# cell (only entries relevant to our 4 datasets kept) to avoid known hangs.
#
# Checkpointed per (dataset, seed): safe to resume if killed/resubmitted.
# ==========================================================================

from _e4_lib import *
from make_mock_cont import generate_make_classification_suite
import os

SYN_DATASET_NAME = "mc_f80_red40_inf24_seed787359109"


def _load_synthetic_d80():
    suite = generate_make_classification_suite(
        n_datasets=30, n_samples=1000, feature_grid=(5, 10, 20, 40, 80),
        redundancy_grid=(0.0, 0.25, 0.5, 0.75), informative_frac=0.3,
        class_sep=1.2, flip_y=0.1, random_state=42,
    )
    ds_map = {ds.name: ds for ds in suite}
    assert SYN_DATASET_NAME in ds_map, f"{SYN_DATASET_NAME!r} not found in regenerated suite."
    return ds_map[SYN_DATASET_NAME]


DATASET_LOADERS = {
    SYN_DATASET_NAME: _load_synthetic_d80,
    "breast_cancer": lambda: load_many(["breast_cancer"], openml_version=2)[0],
    "credit-g": lambda: load_many(["credit-g"], openml_version=2)[0],
    "heloc": lambda: load_many(["heloc"], openml_version=2)[0],
}
DATASETS = list(DATASET_LOADERS.keys())
SEEDS = [1]
N_FACTUAL = 15
METHODS = ("pace", "nice_base", "nice_spars", "mcce", "dice")

# Subset of tabpfn_cf5.ipynb's historical _nice_base_skip / _nice_spars_skip
# relevant to these 4 dataset names (avoids known TabPFN+NICE hangs).
NICE_BASE_SKIP = set()
NICE_SPARS_SKIP = {
    ("TabPFN", "heloc"),
}

ROWS_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/e4_tabpfn_cost.joblib"
CACHE_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/e4_tabpfn_cost_cache_builds.joblib"

if __name__ == "__main__":
    import torch
    print("cuda available:", torch.cuda.is_available(), flush=True)

    t_start = time.perf_counter()

    all_rows, all_cache = [], []
    done_pairs = set()
    if os.path.exists(ROWS_PATH):
        df_existing = joblib.load(ROWS_PATH)
        if isinstance(df_existing, pd.DataFrame) and len(df_existing):
            all_rows.append(df_existing)
            done_pairs = set(zip(df_existing["dataset"], df_existing.get("seed", 1)))
            print(f"Resuming: {len(done_pairs)} (dataset, seed) pair(s) already done.", flush=True)
    if os.path.exists(CACHE_PATH):
        df_cache_existing = joblib.load(CACHE_PATH)
        if isinstance(df_cache_existing, pd.DataFrame) and len(df_cache_existing):
            all_cache.append(df_cache_existing)

    n_total_pairs = len(DATASETS) * len(SEEDS)
    n_pair = 0

    for dataset_name in DATASETS:
        for seed in SEEDS:
            n_pair += 1
            if (dataset_name, seed) in done_pairs:
                print(f"[{n_pair}/{n_total_pairs}] Skipping {dataset_name} seed={seed} (checkpointed)", flush=True)
                continue

            t_pair = time.perf_counter()
            print(f"[{n_pair}/{n_total_pairs}] {dataset_name} seed={seed} ...", flush=True)

            ds_in = DATASET_LOADERS[dataset_name]()
            clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
            X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
                prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)

            models = fit_models(X_tr, y_tr, seed=seed, include_tabpfn=True)
            dice_spec["models"] = fit_models_for_dice(
                X_tr_raw, y_tr, pre, outcome_name=dice_spec["outcome_name"], seed=seed, models_sc=models
            )
            dice_spec["opt_method"] = choose_dice_method(ds_in.name)

            rng = np.random.default_rng(seed)
            factual_indices = rng.choice(X_te.shape[0], size=min(N_FACTUAL, X_te.shape[0]), replace=False)

            df, cache_df = run_benchmark_instrumented(
                X_tr_sc=X_tr, X_te_sc=X_te, y_tr=y_tr, y_te=y_te,
                models=models,
                factual_indices=factual_indices,
                methods=METHODS,
                seed=seed,
                feature_info=feature_info,
                pre=pre,
                X_tr_raw=X_tr_raw,
                X_te_raw=X_te_raw,
                dice_spec=dice_spec,
                dataset_name=ds_in.name,
                nice_base_skip=NICE_BASE_SKIP,
                nice_spars_skip=NICE_SPARS_SKIP,
            )
            df["seed"] = seed
            df["dice_opt_method"] = dice_spec["opt_method"]
            df["n_cols_model"] = meta.get("n_cols_model")
            df["n_rows"] = meta.get("n_rows")
            cache_df["seed"] = seed

            all_rows.append(df)
            all_cache.append(cache_df)
            done_pairs.add((dataset_name, seed))

            joblib.dump(pd.concat(all_rows, ignore_index=True), ROWS_PATH)
            joblib.dump(pd.concat(all_cache, ignore_index=True), CACHE_PATH)

            dt_pair = time.perf_counter() - t_pair
            print(f"  -> {len(df)} rows, {int(df['ok'].sum())} ok, took {dt_pair:.1f}s. "
                  f"({len(done_pairs)}/{n_total_pairs} pairs done)", flush=True)

    t_total = time.perf_counter() - t_start
    print(f"\nDONE. Total wall clock: {t_total/3600:.2f}h", flush=True)
    print(f"Wrote {ROWS_PATH} and {CACHE_PATH}", flush=True)
