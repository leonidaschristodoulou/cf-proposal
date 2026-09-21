# ==========================================================================
# Pilot for a1_extra_datasets_ablation.py: 1 seed, 5 factuals, 2 configs
# (baseline + loo_no_anchor), on TabPFN for both new datasets. Checks the
# synthetic d80 dataset loader reproduces the expected shape/name, and that
# both dataset types run cleanly through the same PACE ablation pipeline.
# Does not touch a1_extra_datasets_ablation.joblib.
# ==========================================================================

import time

import a1_extra_datasets_ablation as a1x

a1x.CONFIGS = a1x.CONFIGS[:2]  # baseline + loo_no_anchor only
a1x.N_FACTUAL = 5
a1x.N_FACTUAL_POOL = 100

if __name__ == "__main__":
    syn_ds = a1x._load_synthetic_d80()
    print(f"synthetic dataset: name={syn_ds.name}, shape={syn_ds.X.shape}, "
          f"class_balance={__import__('numpy').bincount(syn_ds.y)}")
    assert syn_ds.X.shape[1] == 80, f"expected 80 features, got {syn_ds.X.shape[1]}"

    for dataset_name in a1x.DATASETS:
        t0 = time.perf_counter()
        rows = a1x.run_triple(dataset_name, "TabPFN", seed=1)
        dt = time.perf_counter() - t0
        n_ok = sum(1 for r in rows if r["ok"])
        print(f"\n{dataset_name}/TabPFN: {len(rows)} rows ({a1x.N_FACTUAL} factuals x {len(a1x.CONFIGS)} configs), "
              f"{n_ok} ok, took {dt:.1f}s total ({dt/len(rows):.3f}s/run avg).")
        for r in rows[:4]:
            print(r)
