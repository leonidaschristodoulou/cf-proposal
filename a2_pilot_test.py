# ==========================================================================
# A2 pilot: feasibility/timing check before committing to the full sbatch
# run. 1 seed, 5 factuals, 4 configs (one per corner: default M, default s,
# a gamma variant, a surrogate variant) on breast_cancer/TabPFN (the most
# expensive cell in the A2 grid). Does not touch
# a2_hyperparameter_ablation.joblib.
# ==========================================================================

import time

import a2_hyperparameter_ablation as a2

# Keep one config from each of a few axes, including the priciest one (M=10000).
_keep_labels = {"M=2000", "M=10000", "gamma=0.5", "surrogate=none_uniform", "order=l2_l0"}
a2.CONFIGS = [c for c in a2.CONFIGS if c["label"] in _keep_labels]
a2.N_FACTUAL = 5
a2.N_FACTUAL_POOL = 100

if __name__ == "__main__":
    t0 = time.perf_counter()
    rows = a2.run_triple("breast_cancer", "TabPFN", seed=1)
    dt = time.perf_counter() - t0
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n{len(rows)} rows ({a2.N_FACTUAL} factuals x {len(a2.CONFIGS)} configs), "
          f"{n_ok} ok, took {dt:.1f}s total ({dt/len(rows):.3f}s/run avg).")
    for r in rows:
        print(r)
