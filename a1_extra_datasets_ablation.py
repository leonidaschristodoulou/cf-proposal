# ==========================================================================
# A1 follow-up: extend the proposal-mechanism ablation to 2 additional
# "hard, high-dimensional, continuous, few good donors" datasets, to check
# whether breast_cancer's mechanism-importance pattern (anchor >> boundary >
# noise, with a real anchor/boundary l0/l2 tradeoff) generalizes or was
# dataset-specific.
#
# Motivation: in the original 4-dataset A1 run, checking which instances
# succeed in the full baseline but fail once a mechanism is removed (i.e.
# that mechanism was the SOLE route) showed breast_cancer accounts for
# 87.5% of noise's essential instances (14/16), 69% of anchor's (74/108),
# and most of boundary's (outside australian/TabPFN) -- a real N~1
# generalizability gap for the paper's mechanism-importance claim.
#
# Datasets:
#   - heloc (real, mixed features) -- already flagged elsewhere (L2) as a
#     hard TabPFN cell (10-11% incompleteness in the main benchmark).
#   - mc_f80_red40_inf24_seed787359109 (synthetic, 80 continuous features,
#     40 redundant, class_sep=1.2) -- the exact d=80 case already used in
#     completness.ipynb's Appendix B synthetic-completeness investigation,
#     regenerated via the identical generate_make_classification_suite call
#     (n_datasets=30, n_samples=1000, feature_grid=(5,10,20,40,80),
#     redundancy_grid=(0,0.25,0.5,0.75), informative_frac=0.3, class_sep=1.2,
#     flip_y=0.1, random_state=42) so it's the identical dataset instance,
#     not just a same-shape resample.
#
# Reuses the exact same CONFIGS (19: baseline + 3 leave-one-out + 15 mixture
# grid), classifiers (LR/RF/XGB/TabPFN), seeds ([1,3,5]), and factuals-per-cell
# (30) as a1_proposal_mechanism_ablation.py -- imported directly from that
# module (not copy-pasted) so the two runs can't drift out of sync.
#
# Writes to a NEW file (a1_extra_datasets_ablation.joblib) -- does not touch
# a1_mixture_ablation.joblib. Merging the two is a separate, explicit step.
# ==========================================================================

import os
import time

from _a1a2_lib import *  # noqa: F401,F403
import a1_proposal_mechanism_ablation as a1base
from make_mock_cont import generate_make_classification_suite

SYN_DATASET_NAME = 'mc_f80_red40_inf24_seed787359109'

MODELS = a1base.MODELS
SEEDS = a1base.SEEDS
N_FACTUAL = a1base.N_FACTUAL
N_FACTUAL_POOL = a1base.N_FACTUAL_POOL
CONFIGS = a1base.CONFIGS
FIXED = a1base.FIXED
GAMMA_DEFAULT = a1base.GAMMA_DEFAULT


def _load_synthetic_d80():
    suite = generate_make_classification_suite(
        n_datasets=30, n_samples=1000, feature_grid=(5, 10, 20, 40, 80),
        redundancy_grid=(0.0, 0.25, 0.5, 0.75), informative_frac=0.3,
        class_sep=1.2, flip_y=0.1, random_state=42,
    )
    ds_map = {ds.name: ds for ds in suite}
    assert SYN_DATASET_NAME in ds_map, (
        f"{SYN_DATASET_NAME!r} not found in regenerated suite. "
        f"Available 80-feat names: {[n for n in ds_map if 'f80' in n]}"
    )
    return ds_map[SYN_DATASET_NAME]


DATASET_LOADERS = {
    'heloc': lambda: load_many(['heloc'], openml_version=2)[0],
    SYN_DATASET_NAME: _load_synthetic_d80,
}
DATASETS = list(DATASET_LOADERS.keys())

CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/a1_extra_datasets_ablation.joblib"


def run_triple(dataset_name, model_name, seed):
    ds_in = DATASET_LOADERS[dataset_name]()
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
            rs = stable_int_seed(seed, model_name, dataset_name, "a1extra", cfg["name"], int(idx))
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
