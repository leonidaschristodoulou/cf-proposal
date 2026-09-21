# ==========================================================================
# A2: hyperparameter ablation (Workstream 5, PACE_revision_action_list.md).
#
# One-at-a-time sweeps around the Table 1 defaults, on 2 representative
# cells (not the full A1 cross -- see PACE_revision_action_list.md's A2 spec
# and the agreed workstream-5 scope): breast_cancer/TabPFN (the hard,
# continuous-only, high-d case) and credit-g/RF (a mixed-data, non-TabPFN
# cell). Seeds [1,3,5], 30 factuals/cell, same sampling convention as A1.
#
# Axes (7), each swept independently around the shared default
# (M=2000, s=8, base_sigma=0.25, sigma_max=1.0, alpha_range=(0.5,1.0),
# gamma=1.0, guidance=LR, order="l0_l2"); mixture weights pi_a/pi_b/pi_n are
# NOT touched here (that's A1's axis) -- always left at the Table 1 default
# (0.50, 0.35, 0.15):
#   M            in {500, 1000, 2000, 5000, 10000}
#   s            in {2, 4, 8, 16}
#   sigma_base   in {0.1, 0.25, 0.5, 1.0}          (sigma_max held at 1.0)
#   alpha_range  in {(0.5,1.0), (0.25,1.0), (0.0,1.0)}
#   gamma        in {0.5, 1, 2}                     (importance sharpening)
#   surrogate    in {LR, RF, none/uniform}          (none/uniform = gamma=0
#                                                     with LR-driven anchor/
#                                                     boundary pools -- see
#                                                     _a1a2_lib.py docstring)
#   lex_order    in {l0_l2, l2_l0}                  (stageC_select_best's new
#                                                     "order" param)
# 24 (axis, value) configs total, each including its own default point for
# a self-contained per-axis table/plot.
#
# gamma and surrogate are Stage A (cache-level) parameters -- a fresh cache
# is only built per distinct (guidance_model, gamma) pair actually needed
# (5 per dataset/model/seed, not 24), cached in-memory within one triple's run.
#
# Checkpointed per (dataset, model, seed) triple. Writes a NEW file
# (a2_hyperparameter_ablation.joblib) -- does not touch any existing joblib
# cache.
# ==========================================================================

import os
import time

from _a1a2_lib import *  # noqa: F401,F403

CELLS = [("breast_cancer", "TabPFN"), ("credit-g", "RF")]
SEEDS = [1, 3, 5]
N_FACTUAL = 30
N_FACTUAL_POOL = 100

DEFAULTS = dict(n_candidates=2000, max_changed_features=8, base_sigma=0.25, sigma_max=1.0,
                 alpha_range=(0.5, 1.0), gamma=1.0, guidance="LR", order="l0_l2")
MIXTURE_DEFAULT = dict(frac_anchor_mix=0.50, frac_boundary_mix=0.35, frac_guided_noise=0.15)


def _axis_configs():
    configs = []

    for m in [500, 1000, 2000, 5000, 10000]:
        configs.append(dict(axis="M", label=f"M={m}", value=m, n_candidates=m))
    for s in [2, 4, 8, 16]:
        configs.append(dict(axis="s", label=f"s={s}", value=s, max_changed_features=s))
    for sb in [0.1, 0.25, 0.5, 1.0]:
        configs.append(dict(axis="sigma_base", label=f"sigma_base={sb}", value=sb, base_sigma=sb))
    for ar in [(0.5, 1.0), (0.25, 1.0), (0.0, 1.0)]:
        configs.append(dict(axis="alpha_range", label=f"alpha_range={ar}", value=str(ar), alpha_range=ar))
    for g in [0.5, 1.0, 2.0]:
        configs.append(dict(axis="gamma", label=f"gamma={g}", value=g, gamma=g))
    for surr in ["LR", "RF", "none_uniform"]:
        if surr == "none_uniform":
            configs.append(dict(axis="surrogate", label="surrogate=none_uniform", value=surr,
                                 guidance="LR", gamma=0.0))
        else:
            configs.append(dict(axis="surrogate", label=f"surrogate={surr}", value=surr,
                                 guidance=surr, gamma=1.0))
    for order in ["l0_l2", "l2_l0"]:
        configs.append(dict(axis="lex_order", label=f"order={order}", value=order, order=order))

    return configs


CONFIGS = _axis_configs()
assert len(CONFIGS) == 24, f"expected 24 configs, got {len(CONFIGS)}"

CHECKPOINT_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/a2_hyperparameter_ablation.joblib"


def run_triple(dataset_name, model_name, seed):
    ds_in = load_many([dataset_name], openml_version=2)[0]
    clip_q = DICE_CLIP_QUANTILES.get(ds_in.name)
    X_tr, X_te, y_tr, y_te, feature_info, meta, pre, X_tr_raw, X_te_raw, dice_spec = \
        prepare_dataset(ds_in, rng=seed, dice_clip_quantiles=clip_q)

    models = fit_models(X_tr, y_tr, seed=seed, include_tabpfn=(model_name == "TabPFN"))
    pp_target = proba_fn(models[model_name])

    # Memoize caches by (guidance_model_name, gamma) -- only 5 distinct pairs
    # are actually needed across the 24 configs (see module docstring).
    cache_memo = {}

    def get_cache(guidance_name, gamma):
        key = (guidance_name, gamma)
        if key not in cache_memo:
            pp_guidance = proba_fn(models[guidance_name])
            cache_memo[key] = guided_cf_build_cache_ablation(
                X_tr_sc=X_tr, predict_proba_target=pp_target, predict_proba_guidance=pp_guidance,
                feature_info=feature_info, gamma=gamma,
            )
        return cache_memo[key]

    rng = np.random.default_rng(seed)
    factual_pool = rng.choice(X_te.shape[0], size=min(N_FACTUAL_POOL, X_te.shape[0]), replace=False)
    factual_indices = factual_pool[:N_FACTUAL]

    rows = []
    for idx in factual_indices:
        x_f = X_te[int(idx)]
        for cfg in CONFIGS:
            params = dict(DEFAULTS)
            params.update({k: v for k, v in cfg.items() if k not in ("axis", "label", "value")})
            guidance_name = params.pop("guidance")
            gamma = params.pop("gamma")
            cache = get_cache(guidance_name, gamma)

            # lex_order must reuse the SAME candidate draw across both order values --
            # order only changes which already-flipped candidate is picked, so it
            # cannot legitimately change completeness. Every other axis genuinely
            # changes what gets generated, so varying the seed by config value there
            # is fine (and desired, to avoid seed-correlated artifacts).
            seed_value = "shared" if cfg["axis"] == "lex_order" else str(cfg["value"])
            rs = stable_int_seed(seed, model_name, dataset_name, "a2", cfg["axis"], seed_value, int(idx))
            t0 = time.perf_counter()
            x_cf, rep = pace_with_params_ablation(
                x_f_sc=x_f, cache=cache, predict_proba_target=pp_target,
                random_state=rs, **MIXTURE_DEFAULT, **params,
            )
            dt = time.perf_counter() - t0
            status = rep.get("status", "")
            ok = (x_cf is not None) and (status == "ok")
            rows.append({
                "dataset": dataset_name, "model": model_name, "seed": int(seed), "idx": int(idx),
                "param_name": cfg["axis"], "param_value": str(cfg["value"]), "config_label": cfg["label"],
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

    triples = [(d, m, s) for (d, m) in CELLS for s in SEEDS]
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
    print("\n===== FINAL COMPLETENESS BY (dataset, model, param_name, param_value) =====", flush=True)
    print(df_final.groupby(["dataset", "model", "param_name", "param_value"])["ok"]
          .agg(["size", "sum", "mean"]).to_string(), flush=True)
    print(f"\nWrote {CHECKPOINT_PATH}", flush=True)
