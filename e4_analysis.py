# ==========================================================================
# E4 analysis: build tabpfn_cost.csv (target-model calls/rows per method x
# dataset x classifier) and the runtime-vs-target-call-count demonstration
# plot (one panel per classifier, all methods) from e4_tabpfn_cost.joblib.
#
# Colors: reuses cfprop_plots.ipynb's own `COLORS` dict (cell 4) verbatim --
# the canonical method-identity palette already used for every baseline-
# comparison figure in the paper (flip-rate bars, l0-vs-l2 scatter, Pareto
# dominance, the credit-g/breast-cancer qualitative examples). Not
# PROPOSAL_COLORS (that one's categorical dimension is proposal mechanism --
# anchor/boundary/noise -- a different axis entirely; reusing it here would
# have been its own inconsistency).
# ==========================================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = joblib.load("e4_tabpfn_cost.joblib")

MODEL_ORDER = ["LR", "RF", "XGB", "TabPFN"]
METHOD_ORDER = ["pace", "nice_base", "nice_spars", "mcce", "dice"]
METHOD_LABELS = {
    "pace": "PACE", "nice_base": "NICE-base", "nice_spars": "NICE-sparse",
    "mcce": "MCCE", "dice": "DiCE",
}
METHOD_COLORS = {
    "nice_base": "#0072B2",   # blue
    "nice_spars": "#E69F00",  # orange
    "dice": "#D55E00",        # vermillion
    "mcce": "#CC79A7",        # reddish purple
    "pace": "#009E73",        # bluish green
}

# ---------------------------------------------------------------------------
# tabpfn_cost.csv: per (dataset, model, method) summary
# ---------------------------------------------------------------------------
summary = (
    df.groupby(["dataset", "model", "method"])
    .agg(
        n_attempted=("ok", "size"),
        n_ok=("ok", "sum"),
        target_calls_mean=("target_calls", "mean"),
        target_calls_median=("target_calls", "median"),
        target_calls_min=("target_calls", "min"),
        target_calls_max=("target_calls", "max"),
        target_rows_evaluated_mean=("target_rows_evaluated", "mean"),
        time_s_mean=("time_s", "mean"),
        time_s_median=("time_s", "median"),
    )
    .reset_index()
)
summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
summary = summary.sort_values(["dataset", "model", "method"]).reset_index(drop=True)
summary.to_csv("tabpfn_cost.csv", index=False)
print(f"Wrote tabpfn_cost.csv ({len(summary)} rows)")

# Overall (pooled across datasets) per-model x per-method table -- the
# headline numbers for the paper's efficiency-mechanism table.
overall = (
    df.groupby(["model", "method"])
    .agg(
        n_attempted=("ok", "size"),
        target_calls_mean=("target_calls", "mean"),
        target_calls_median=("target_calls", "median"),
        target_rows_evaluated_mean=("target_rows_evaluated", "mean"),
        time_s_mean=("time_s", "mean"),
        time_s_median=("time_s", "median"),
    )
    .reset_index()
)
overall["model"] = pd.Categorical(overall["model"], MODEL_ORDER, ordered=True)
overall["method"] = pd.Categorical(overall["method"], METHOD_ORDER, ordered=True)
overall = overall.sort_values(["model", "method"]).reset_index(drop=True)
overall.to_csv("tabpfn_cost_overall.csv", index=False)
print(f"Wrote tabpfn_cost_overall.csv ({len(overall)} rows)")
print(overall.to_string(index=False))

# ---------------------------------------------------------------------------
# Demonstration plot: runtime vs target-call-count, one panel per classifier
# (2x2 grid; axis labels only on the shared bottom row / left column; each
# classifier named as in-panel text rather than a title; legend inside the
# LR panel, the least visually crowded one).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), sharex=True, sharey=True)
axes_flat = axes.flatten()

for i, (ax, model_name) in enumerate(zip(axes_flat, MODEL_ORDER)):
    sub = df[df["model"] == model_name]
    for method in METHOD_ORDER:
        m = sub[sub["method"] == method]
        if len(m) == 0:
            continue
        # jitter x slightly (integer call counts overplot heavily) for readability
        rng = np.random.default_rng(0)
        x_jit = m["target_calls"].to_numpy() * np.exp(rng.normal(0, 0.03, size=len(m)))
        ax.scatter(
            x_jit, m["time_s"], s=18, alpha=0.55,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 100)
    ax.text(
        0.03, 0.95, model_name, transform=ax.transAxes,
        ha="left", va="top", fontsize=12, fontweight="bold",
    )
    ax.grid(True, which="both", alpha=0.15)

    row, col = divmod(i, 2)
    if row == 1:  # bottom row only
        ax.set_xlabel("target-model calls per CE (log)")
    if col == 0:  # left column only
        ax.set_ylabel("runtime per CE, seconds (log)")

axes_flat[0].legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.85)
fig.tight_layout()
fig.savefig("e4_runtime_vs_calls.pdf", bbox_inches="tight")
print("Wrote e4_runtime_vs_calls.pdf")

# Spearman correlation (calls vs runtime) per classifier, pooled across
# methods/datasets -- the quantitative backing for "the correlation is the
# proof that runtime is governed by call count."
from scipy.stats import spearmanr
print("\nSpearman rank correlation(target_calls, time_s) by classifier (all methods pooled):")
for model_name in MODEL_ORDER:
    sub = df[df["model"] == model_name]
    rho, p = spearmanr(sub["target_calls"], sub["time_s"])
    print(f"  {model_name}: rho={rho:.3f}, p={p:.2e}, n={len(sub)}")
