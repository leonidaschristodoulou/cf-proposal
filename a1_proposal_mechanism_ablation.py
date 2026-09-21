# ==========================================================================
# A1: proposal-mechanism ablation (Workstream 5, PACE_revision_action_list.md).
#
# Two sub-studies on PACE's Stage B mixture weights pi_a/pi_b/pi_n
# (frac_anchor_mix / frac_boundary_mix / frac_guided_noise):
#   1. Leave-one-out: zero one mechanism, keep the other two's *relative*
#      ratio (stageB_generate renormalizes automatically -- see
#      pfn_cf_guided_onehot.py:606-610), report deltas vs the full method.
#   2. Mixture sweep: full simplex grid over {0, 0.25, 0.5, 0.75, 1.0}
#      (15 valid combos summing to 1) plus the Table-1-default baseline
#      (0.50, 0.35, 0.15) -- baseline isn't on the quantized grid, so it's
#      run separately. 19 configs total.
#
# All other PACE hyperparameters are held at Table 1 defaults throughout
# (n_candidates=2000, max_changed_features=8, base_sigma=0.25, sigma_max=1.0,
# alpha_range=(0.5,1.0), gamma=1.0, guidance=LR, order="l0_l2") -- this
# isolates the mixture-weight effect, which is what A1 is about. Guidance is
# always LR for every target model, matching the paper's fixed design (M6).
#
# Datasets: breast_cancer (hard, continuous-only, high-d), blood_transfusion
# (continuous-only, no categoricals), credit-g and australian (mixed,
# categorical) -- the representative subset agreed for workstream 5.
# Models: LR, RF, XGB, TabPFN (full cross, per the agreed A1 scope -- this is
# the ablation that has to span regimes, per M5/Fig. 9's anchors-vs-boundary-
# vs-noise dominance-by-dataset/classifier claim).
# Seeds: [1, 3, 5] (subset of the paper's standard [1,3,5,7,9]).
# Factuals: first 30 of the same rng.choice(n_test, size=100, replace=False)
# draw the main benchmark uses per seed, so these instances overlap with
# output_realdata.joblib's known per-instance PACE outcomes.
#
# Checkpointed per (dataset, model, seed) triple. Writes a NEW file
# (a1_mixture_ablation.joblib) -- does not touch any existing joblib cache.
# ==========================================================================

import itertools
import os
import time

from _a1a2_lib import *  # noqa: F401,F403

DATASETS = ["breast_cancer", "blood_transfusion", "credit-g", "australian"]
MODELS = ["LR", "RF", "XGB", "TabPFN"]
SEEDS = [1, 3, 5]
N_FACTUAL = 30
N_FACTUAL_POOL = 100  # matches the main benchmark's per-seed draw size

BASELINE = dict(name="baseline", frac_anchor_mix=0.50, frac_boundary_mix=0.35, frac_guided_noise=0.15)
LEAVE_ONE_OUT = [
    dict(name="loo_no_anchor",   frac_anchor_mix=0.00, frac_boundary_mix=0.35, frac_guided_noise=0.15),
    dict(name="loo_no_boundary", frac_anchor_mix=0.50, frac_boundary_mix=0.00, frac_guided_noise=0.15),
    dict(name="loo_no_noise",    frac_anchor_mix=0.50, frac_boundary_mix=0.35, frac_guided_noise=0.00),
]


def _mixture_grid():
    vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    configs = []
    for a, b, c in itertools.product(vals, repeat=3):
        if abs(a + b + c - 1.0) < 1e-9:
            configs.append(dict(
                name=f"mix_a{a:.2f}_b{b:.2f}_c{c:.2f}",
                frac_anchor_mix=a, frac_boundary_mix=b, frac_guided_noise=c,
            ))
    return configs


CONFIGS = [BASELINE] + LEAVE_ONE_OUT + _mixture_grid()
assert len(CONFIGS) == 19, f"expected 19 configs, got {len(CONFIGS)}"

# Fixed Table 1 defaults for every other knob (this ablation only varies the mixture).
FIXED = dict(n_candidates=2000, max_changed_features=8, base_sigma=0.25, sigma_max=1.0,
             alpha_range=(0.5, 1.0), order="l0_l2")
GAMMA_DEFAULT = 1.0

CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/a1_mixture_ablation.joblib"


def run_triple(dataset_name, model_name, seed):
    ds_in = load_many([dataset_name], openml_version=2)[0]
    clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)

    models = fit_models(X_tr, y_tr, seed=seed, include_tabpfn=(model_name == "TabPFN"))
    pp_target = proba_fn(models[model_name])
    pp_guidance = proba_fn(models["LR"])

    cache = guided_cf_build_cache_ablation(
        X_tr_sc=X_tr, predict_proba_target=pp_target, predict_proba_guidance=pp_guidance,
        feature_info=feature_info, gamma=GAMMA_DEFAULT,
    )

    rng = np.random.default_rng(seed)
    factual_pool = rng.choice(X_te.shape[0], size=min(N_FACTUAL_POOL, X_te.shape[0]), replace=False)
    factual_indices = factual_pool[:N_FACTUAL]

    rows = []
    for idx in factual_indices:
        x_f = X_te[int(idx)]
        for cfg in CONFIGS:
            rs = stable_int_seed(seed, model_name, dataset_name, "a1", cfg["name"], int(idx))
            t0 = time.perf_counter()
            x_cf, rep = pace_with_params_ablation(
                x_f_sc=x_f, cache=cache, predict_proba_target=pp_target,
                frac_anchor_mix=cfg["frac_anchor_mix"],
                frac_boundary_mix=cfg["frac_boundary_mix"],
                frac_guided_noise=cfg["frac_guided_noise"],
                random_state=rs, **FIXED,
            )
            dt = time.perf_counter() - t0
            status = rep.get("status", "")
            ok = (x_cf is not None) and (status == "ok")
            rows.append({
                "dataset": dataset_name, "model": model_name, "seed": int(seed), "idx": int(idx),
                "config": cfg["name"],
                "frac_anchor_mix": cfg["frac_anchor_mix"],
                "frac_boundary_mix": cfg["frac_boundary_mix"],
                "frac_guided_noise": cfg["frac_guided_noise"],
                "status": status, "ok": bool(ok),
                "l0": rep.get("l0", None), "l2": rep.get("l2", None),
                "p_cf": rep.get("p_cf", None), "n_flip": rep.get("n_flip", None),
                "time_s": float(dt),
            })
    return rows


if __name__ == "__main__":
    all_rows = []
    done_triples = set()
    if os.path.exists(CHECKPOINT_PATH):
        df_existing = joblib.load(CHECKPOINT_PATH)
        if isinstance(df_existing, pd.DataFrame) and len(df_existing):
            all_rows.append(df_existing)
            done_triples = set(zip(df_existing["dataset"], df_existing["model"], df_existing["seed"]))
            print(f"Resuming: {len(done_triples)} triple(s) already done.", flush=True)

    triples = [(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS]
    for i, (dataset_name, model_name, seed) in enumerate(triples, 1):
        if (dataset_name, model_name, seed) in done_triples:
            print(f"[{i}/{len(triples)}] Skipping {dataset_name}/{model_name} seed={seed} (checkpointed)", flush=True)
            continue

        print(f"[{i}/{len(triples)}] {dataset_name}/{model_name} seed={seed} "
              f"({len(CONFIGS)} configs x {N_FACTUAL} factuals = {len(CONFIGS)*N_FACTUAL} runs) ...", flush=True)
        t0 = time.perf_counter()
        rows = run_triple(dataset_name, model_name, seed)
        df_triple = pd.DataFrame(rows)
        all_rows.append(df_triple)
        done_triples.add((dataset_name, model_name, seed))

        df_ckpt = pd.concat(all_rows, ignore_index=True)
        joblib.dump(df_ckpt, CHECKPOINT_PATH)

        n_ok = sum(1 for r in rows if r["ok"])
        dt_triple = time.perf_counter() - t0
        print(f"  -> {len(rows)} rows, {n_ok} ok ({100*n_ok/len(rows):.1f}%), took {dt_triple:.1f}s. "
              f"Checkpoint saved ({len(done_triples)}/{len(triples)} triples, {len(df_ckpt)} rows total).",
              flush=True)

    df_final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    print("\n===== FINAL COMPLETENESS BY (dataset, model, config) =====", flush=True)
    print(df_final.groupby(["dataset", "model", "config"])["ok"].agg(["size", "sum", "mean"]).to_string(), flush=True)
    print(f"\nWrote {CHECKPOINT_PATH}", flush=True)
