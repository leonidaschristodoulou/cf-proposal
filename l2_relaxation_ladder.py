# ==========================================================================
# L2: broaden the completeness-recovery relaxation ladder beyond the single
# Breast-Cancer/TabPFN case (completness.ipynb), to Australian/ILPD/HELOC +
# TabPFN. Faithfully ported from completness.ipynb cells 1,3,4,6,10,11,13,
# 15,17,19,21,24,27,28 (the real-data relaxation-ladder pipeline, not the
# synthetic d=80 section) -- same helper functions, same 6 configs, same
# seeds/factual-sampling logic, only TARGET_DATASET/TARGET_MODEL vary.
#
# Usage: python l2_relaxation_ladder.py <dataset_name> <model_name>
# Writes: l2_sweep_{dataset}_{model}.joblib (full per-instance sweep results),
#         real_recovery_rate_{dataset}_{model}.pdf,
#         real_recovery_cost_{dataset}_{model}.pdf,
#         real_l0l2_baseline_{dataset}_{model}.pdf
# Does not touch output_realdata.joblib or any existing completness.ipynb
# output file.
# ==========================================================================

from __future__ import annotations
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

from sklearn.model_selection import train_test_split

sys.path.insert(0, "/nvme/h/lchristodoulou/pace/cf-proposal/")
from pfn_cf_guided_onehot import (
    stageA_build, stageA_target_anchors_from_p, stageB_generate, stageC_select_best, FeatureInfo
)
from get_real_datasets import load_many, ImmutableSpec
from preprocessing import (
    make_preprocessor, build_feature_info_from_pre,
    safe_immutable_num, safe_immutable_cat,
)

TARGET_DATASET = sys.argv[1]
TARGET_MODEL = sys.argv[2]
SEEDS = (1, 3, 5, 7, 9)
N_FACTUAL = 100
RESULTS_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/output_realdata.joblib"

print(f"===== L2 relaxation ladder: {TARGET_DATASET} / {TARGET_MODEL} =====", flush=True)


# ── Helpers replicated from completness.ipynb (originally from tabpfn_cf5.ipynb) ──

@dataclass
class GuidedCFCache:
    X_tr_sc: np.ndarray
    predict_proba_guidance: Any
    feature_info: Optional[FeatureInfo]
    p_train: np.ndarray
    A: Any
    anchors_by_class: Dict[int, np.ndarray]
    anchors_diag_by_class: Dict[int, Dict[str, Any]]


def guided_cf_build_cache(
    *, X_tr_sc, predict_proba_target, predict_proba_guidance=None,
    stageA_random_state=0, anchor_k=250, anchors_random_state=1, feature_info=None,
):
    if predict_proba_guidance is None:
        predict_proba_guidance = predict_proba_target
    p_train = predict_proba_guidance(X_tr_sc).astype(np.float64)
    A = stageA_build(
        X_train=X_tr_sc, predict_proba_fn=predict_proba_guidance,
        random_state=stageA_random_state, feature_info=feature_info,
    )
    anchors_by_class, anchors_diag_by_class = {}, {}
    for y_desired in (0, 1):
        anchors, diag = stageA_target_anchors_from_p(
            p_train=p_train, y_desired=y_desired,
            anchor_k=anchor_k, random_state=anchors_random_state,
        )
        anchors_by_class[y_desired] = anchors
        anchors_diag_by_class[y_desired] = diag
    return GuidedCFCache(
        X_tr_sc=X_tr_sc, predict_proba_guidance=predict_proba_guidance,
        feature_info=feature_info, p_train=p_train, A=A,
        anchors_by_class=anchors_by_class, anchors_diag_by_class=anchors_diag_by_class,
    )


def pace_with_params(
    *, x_f_sc, cache, predict_proba_target, n_candidates=2000, max_changed_features=8,
    base_sigma=0.25, sigma_max=1.0, alpha_range=(0.5, 1.0),
    frac_anchor_mix=0.50, frac_boundary_mix=0.35, frac_guided_noise=0.15, random_state=0,
):
    p_f = float(predict_proba_target(x_f_sc[None, :])[0])
    y_desired = 1 - int(p_f >= 0.5)
    anchors = cache.anchors_by_class[y_desired]
    if anchors.size == 0:
        return None, {"status": "no_anchors", "n_flip": 0, "n_candidates": 0}
    C_sc, meta = stageB_generate(
        x_f=x_f_sc, X_train=cache.X_tr_sc, predict_proba_fn=cache.predict_proba_guidance,
        boundary_idx=cache.A.boundary_idx, anchors_idx=anchors,
        feature_weights=cache.A.feature_weights, unit_weights=getattr(cache.A, "unit_weights", None),
        feature_info=cache.feature_info, n_candidates=n_candidates,
        max_changed_features=max_changed_features, base_sigma=base_sigma, sigma_max=sigma_max,
        alpha_range=alpha_range, frac_anchor_mix=frac_anchor_mix, frac_boundary_mix=frac_boundary_mix,
        frac_guided_noise=frac_guided_noise, random_state=random_state,
    )
    x_cf_sc, rep = stageC_select_best(
        x_f=x_f_sc, C=C_sc, sources=meta["sources"], predict_proba_fn=predict_proba_target,
        y_desired=y_desired, feature_info=cache.feature_info,
    )
    return x_cf_sc, rep


def fit_models(X_tr_sc, y_tr, seed=0, include_tabpfn=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    models = {}
    models["LR"] = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=seed).fit(X_tr_sc, y_tr)
    models["XGB"] = XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed, n_jobs=-1,
    ).fit(X_tr_sc, y_tr)
    models["RF"] = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1).fit(X_tr_sc, y_tr)
    if include_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            models["TabPFN"] = TabPFNClassifier(device="auto").fit(X_tr_sc, y_tr)
        except Exception as e:
            print(f"TabPFN skipped: {e}")
    return models


def proba_fn(model):
    return lambda X: model.predict_proba(X)[:, 1]


def stable_int_seed(*parts) -> int:
    s = "|".join(map(str, parts)).encode("utf-8")
    return zlib.crc32(s) & 0xFFFFFFFF


def prepare_for_completeness(ds_in, *, seed=0, test_size=0.25):
    X_df = ds_in.X_df
    y = np.asarray(ds_in.y).astype(int)
    num_cols, cat_cols = list(ds_in.num_cols), list(ds_in.cat_cols)
    idx = np.arange(len(X_df))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    X_tr_raw = X_df.iloc[idx_tr].copy()
    X_te_raw = X_df.iloc[idx_te].copy()
    y_tr, y_te = y[idx_tr], y[idx_te]
    pre = make_preprocessor(num_cols, cat_cols, ohe_sparse=False)
    pre.fit(X_tr_raw)
    X_tr_sc = np.asarray(pre.transform(X_tr_raw), dtype=np.float64)
    X_te_sc = np.asarray(pre.transform(X_te_raw), dtype=np.float64)
    immutables = getattr(ds_in, "immutables", ImmutableSpec.empty())
    if len(cat_cols) == 0 and not (immutables.num or immutables.cat):
        feature_info = None
    else:
        num_idx, cat_groups = build_feature_info_from_pre(pre, num_cols, cat_cols)
        feature_info = FeatureInfo(
            num_idx=num_idx, cat_groups=cat_groups,
            immutable_num=safe_immutable_num(num_cols, num_idx, immutables.num),
            immutable_cat=safe_immutable_cat(cat_cols, immutables.cat),
        )
    return X_tr_sc, X_te_sc, y_tr, y_te, feature_info, pre, num_cols, cat_cols


# ── 1. Load results, identify failures for this target ──

df_all = joblib.load(RESULTS_PATH)
pace = df_all[df_all["method"] == "pace"].copy()
failed = pace[pace["status"] != "ok"].copy()

target_failed = failed[
    (failed["dataset"] == TARGET_DATASET) & (failed["model"] == TARGET_MODEL)
][["seed", "idx", "p_f", "status"]].copy()

print(f"Failed instances for {TARGET_DATASET}/{TARGET_MODEL}: {len(target_failed)}", flush=True)
print(target_failed.groupby("seed").size().to_string(), flush=True)
assert len(target_failed) > 0, "no failed instances for this target -- nothing to recover"

ds_list = load_many([TARGET_DATASET], openml_version=2)
ds = ds_list[0]
print(f"Dataset: {ds.name}, shape: {ds.X_df.shape}", flush=True)


# ── 2. Sweep configs (identical to completness.ipynb) ──

sweep_configs = [
    dict(name="A_baseline",    n_candidates=2000,  max_changed_features=8,  base_sigma=0.25, sigma_max=1.0, alpha_range=(0.5, 1.0)),
    dict(name="B_more_cands",  n_candidates=10000, max_changed_features=8,  base_sigma=0.25, sigma_max=1.0, alpha_range=(0.5, 1.0)),
    dict(name="C_more_feats",  n_candidates=2000,  max_changed_features=20, base_sigma=0.25, sigma_max=1.0, alpha_range=(0.5, 1.0)),
    dict(name="D_wider_sigma", n_candidates=2000,  max_changed_features=8,  base_sigma=0.75, sigma_max=3.0, alpha_range=(0.5, 1.0)),
    dict(name="E_wider_alpha", n_candidates=2000,  max_changed_features=8,  base_sigma=0.25, sigma_max=1.0, alpha_range=(0.1, 1.0)),
    dict(name="F_aggressive",  n_candidates=10000, max_changed_features=20, base_sigma=0.75, sigma_max=3.0, alpha_range=(0.1, 1.0)),
]

# ── 3. Run the sweep ──

sweep_rows = []
for seed in SEEDS:
    seed_failed = target_failed[target_failed["seed"] == seed]
    if len(seed_failed) == 0:
        continue
    print(f"\n-- seed={seed}  ({len(seed_failed)} failed instances) --", flush=True)

    X_tr_sc, X_te_sc, y_tr, y_te, feature_info, pre, num_cols, cat_cols = \
        prepare_for_completeness(ds, seed=seed)
    models = fit_models(X_tr_sc, y_tr, seed=seed, include_tabpfn=True)
    target_model = models[TARGET_MODEL]
    guidance_model = models["LR"]
    pp_target = proba_fn(target_model)
    pp_guidance = proba_fn(guidance_model)

    cache = guided_cf_build_cache(
        X_tr_sc=X_tr_sc, predict_proba_target=pp_target, predict_proba_guidance=pp_guidance,
        stageA_random_state=0, anchor_k=250, anchors_random_state=1, feature_info=feature_info,
    )

    rng = np.random.default_rng(seed)
    factual_indices = rng.choice(X_te_sc.shape[0], size=min(N_FACTUAL, X_te_sc.shape[0]), replace=False)
    failed_te_indices = set(seed_failed["idx"].tolist())

    for te_idx in sorted(failed_te_indices):
        x_f = X_te_sc[te_idx]
        p_f_val = float(pp_target(x_f[None, :])[0])
        for cfg in sweep_configs:
            rs = stable_int_seed(seed, TARGET_MODEL, "guided", int(te_idx))
            t0 = time.perf_counter()
            x_cf, rep = pace_with_params(
                x_f_sc=x_f, cache=cache, predict_proba_target=pp_target,
                n_candidates=cfg["n_candidates"], max_changed_features=cfg["max_changed_features"],
                base_sigma=cfg["base_sigma"], sigma_max=cfg["sigma_max"], alpha_range=cfg["alpha_range"],
                random_state=rs,
            )
            elapsed = time.perf_counter() - t0
            sweep_rows.append({
                "seed": seed, "idx": te_idx, "p_f": p_f_val, "config": cfg["name"],
                "n_candidates": cfg["n_candidates"], "max_changed_features": cfg["max_changed_features"],
                "base_sigma": cfg["base_sigma"], "found": rep.get("status") == "ok",
                "status": rep.get("status", ""), "l0": rep.get("l0", np.nan), "l2": rep.get("l2", np.nan),
                "p_cf": rep.get("p_cf", np.nan), "n_flip": rep.get("n_flip", 0), "time_s": elapsed,
            })
    print(f"   done ({len(seed_failed)} x {len(sweep_configs)} = {len(seed_failed)*len(sweep_configs)} runs)", flush=True)

sweep_df = pd.DataFrame(sweep_rows)
print(f"\nTotal sweep runs: {len(sweep_df)}", flush=True)

tag = f"{TARGET_DATASET}_{TARGET_MODEL}"
joblib.dump(sweep_df, f"l2_sweep_{tag}.joblib")


# ── 4. Recovery rate per config ──

recovery = (
    sweep_df.groupby("config")["found"]
    .agg(recovered="sum", total="count")
    .assign(recovery_rate=lambda d: d["recovered"] / d["total"])
    .reindex([c["name"] for c in sweep_configs])
)
print("\n===== RECOVERY RATE =====", flush=True)
print(recovery.to_string(), flush=True)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(sweep_configs))
bars = ax.bar(x, recovery["recovery_rate"] * 100, color="#009E73", edgecolor="k")
ax.bar_label(bars, fmt="%.0f%%", padding=3)
ax.set_xticks(x)
ax.set_xticklabels([c["name"] for c in sweep_configs], rotation=20, ha="right")
ax.set_ylabel("Recovery rate (%)")
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"real_recovery_rate_{tag}.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)


# ── 5. Cost when recovered ──

found_df = sweep_df[sweep_df["found"]].copy()
cost_summary = (
    found_df.groupby("config")
    .agg(l0_median=("l0", "median"), l0_mean=("l0", "mean"),
         l2_median=("l2", "median"), l2_mean=("l2", "mean"), n=("l0", "count"))
    .reindex([c["name"] for c in sweep_configs])
)
print("\n===== COST WHEN RECOVERED =====", flush=True)
print(cost_summary.to_string(), flush=True)

cfg_names = [c["name"] for c in sweep_configs]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, label in zip(axes, ["l0", "l2"], ["L0 (changed units)", "L2 distance"]):
    data_per_cfg = [found_df[found_df["config"] == name][metric].dropna().values for name in cfg_names]
    bp = ax.boxplot(data_per_cfg, labels=cfg_names, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#009E73")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"real_recovery_cost_{tag}.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)


# ── 6. Not recovered even by the most aggressive config ──

aggressive_config = "F_aggressive"
not_recovered = sweep_df[(sweep_df["config"] == aggressive_config) & (~sweep_df["found"])]
print(f"\nStill not recovered under '{aggressive_config}': {len(not_recovered)} instances", flush=True)


# ── 7. Incremental gain per config ──

cfg_order = [c["name"] for c in sweep_configs]
first_recovery = (
    sweep_df[sweep_df["found"]]
    .assign(cfg_rank=lambda d: d["config"].map({n: i for i, n in enumerate(cfg_order)}))
    .sort_values("cfg_rank")
    .groupby(["seed", "idx"], as_index=False)
    .first()[["seed", "idx", "config", "l0", "l2", "p_f"]]
)
incremental = first_recovery.groupby("config").size().reindex(cfg_order).fillna(0).astype(int)
print("\n===== INCREMENTAL GAIN (instances first recovered by each config) =====", flush=True)
print(incremental.to_string(), flush=True)


# ── 8. Summary table ──

summary = sweep_df.groupby("config").apply(
    lambda g: pd.Series({
        "recovery_rate": g["found"].mean(), "n_recovered": g["found"].sum(),
        "l0_median": g.loc[g["found"], "l0"].median(), "l2_median": g.loc[g["found"], "l2"].median(),
        "time_s_mean": g["time_s"].mean(), "n_candidates": g["n_candidates"].iloc[0],
        "max_feats": g["max_changed_features"].iloc[0], "base_sigma": g["base_sigma"].iloc[0],
    })
).reindex(cfg_order)
print("\n===== SUMMARY =====", flush=True)
print(summary.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)


# ── 9. L0 vs L2: PACE (C_more_feats) vs NICE/DiCE/MCCE on the same originally-failed instances (L1's method) ──

COLORS = {"nice_base": "#0072B2", "nice_spars": "#E69F00", "dice": "#D55E00", "mcce": "#CC79A7", "pace": "#009E73"}
_markers = ["o", "s", "D", "^", "v"]
_all_methods = ["dice", "nice_base", "nice_spars", "mcce", "pace"]
METHOD_MARKER = {m: _markers[i] for i, m in enumerate(_all_methods)}

pace_mf = sweep_df[(sweep_df["config"] == "C_more_feats") & sweep_df["found"]].copy()
pace_mf = pace_mf.assign(method="pace")
target_pairs_set = set(zip(pace_mf["seed"], pace_mf["idx"]))

other_raw = df_all[
    (df_all["dataset"] == TARGET_DATASET) & (df_all["model"] == TARGET_MODEL)
    & (df_all["method"].isin(["nice_base", "nice_spars", "dice", "mcce"]))
].copy()
other_ok = other_raw[
    other_raw.apply(lambda r: (r["seed"], r["idx"]) in target_pairs_set, axis=1)
].dropna(subset=["l0", "l2"])

print("\n===== L1-STYLE BASELINE COMPARISON (on PACE's originally-failed, now-recovered instances) =====", flush=True)
for m in ["dice", "nice_base", "nice_spars", "mcce"]:
    sub = other_ok[other_ok["method"] == m]
    n_total = (other_raw["method"] == m).sum()
    if len(sub):
        print(f"{m}: {len(sub)}/{n_total} succeeded on these instances, "
              f"median l0={sub['l0'].median():.2f}, mean l0={sub['l0'].mean():.2f}", flush=True)
    else:
        print(f"{m}: 0/{n_total} succeeded on these instances", flush=True)
print(f"pace (C_more_feats, relaxed): {len(pace_mf)}/{len(target_failed)} succeeded, "
      f"median l0={pace_mf['l0'].median():.2f}, mean l0={pace_mf['l0'].mean():.2f}", flush=True)

plot_methods = [
    ("dice", "DiCE", other_ok[other_ok["method"] == "dice"]),
    ("nice_base", "NICE (base)", other_ok[other_ok["method"] == "nice_base"]),
    ("nice_spars", "NICE (sparse)", other_ok[other_ok["method"] == "nice_spars"]),
    ("mcce", "MCCE", other_ok[other_ok["method"] == "mcce"]),
    ("pace", "PACE (more feats)", pace_mf),
]
_log_offsets = np.linspace(-0.08, 0.08, len(plot_methods))
_method_offset = {key: _log_offsets[i] for i, (key, _, __) in enumerate(plot_methods)}

fig, ax = plt.subplots(figsize=(9, 5))
for key, label, sub in plot_methods:
    if sub.empty:
        continue
    x_mult = np.exp(_method_offset[key])
    ax.scatter(sub["l0"].values * x_mult, sub["l2"].values, color=COLORS[key],
               marker=METHOD_MARKER[key], s=10, alpha=0.8, label=f"{label} (n={len(sub)})")
ax.set_xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))
ax.set_xlabel(r"$\ell_0$", fontsize=11)
ax.set_ylabel(r"$\ell_2$", fontsize=11)
ax.tick_params(axis="both", which="both", width=0.8, labelsize=11)
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=False)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")
fig.tight_layout()
plt.savefig(f"real_l0l2_baseline_{tag}.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nDone. Outputs: l2_sweep_{tag}.joblib, real_recovery_rate_{tag}.pdf, "
      f"real_recovery_cost_{tag}.pdf, real_l0l2_baseline_{tag}.pdf", flush=True)
