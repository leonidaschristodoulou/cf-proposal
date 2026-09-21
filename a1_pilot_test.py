# ==========================================================================
# A1 pilot: feasibility/timing check before committing to the full sbatch
# run. 1 seed, 5 factuals, 2 configs (baseline + one leave-one-out) on
# breast_cancer/TabPFN (the most expensive cell in the A1 grid). Does not
# touch a1_mixture_ablation.joblib. Mirrors the bounded-test-before-full-run
# pattern used throughout REVISION_EVIDENCE_LOG.md (e.g. e5b_test_dice_tabpfn.py).
# ==========================================================================

import time

import a1_proposal_mechanism_ablation as a1

a1.CONFIGS = a1.CONFIGS[:2]  # baseline + loo_no_anchor only
a1.N_FACTUAL = 5
a1.N_FACTUAL_POOL = 100

if __name__ == "__main__":
    t0 = time.perf_counter()
    rows = a1.run_triple("breast_cancer", "TabPFN", seed=1)
    dt = time.perf_counter() - t0
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"\n{len(rows)} rows ({a1.N_FACTUAL} factuals x {len(a1.CONFIGS)} configs), "
          f"{n_ok} ok, took {dt:.1f}s total ({dt/len(rows):.3f}s/run avg).")
    for r in rows:
        print(r)
