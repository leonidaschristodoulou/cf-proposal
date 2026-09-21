# ==========================================================================
# E1b: scaling analysis quantifying why population-based evolutionary CF
# search (NSGA-II/MIES, as in MOC) is architecturally mismatched to
# expensive in-context predictors like TabPFN -- the quantified argument
# that replaces the declined MOC benchmark (E1's decision).
#
# Pure analysis over E4's already-collected data (e4_tabpfn_cost.joblib) --
# no new run needed, per the spec's "optional empirical anchor... no MOC
# integration necessary": DiCE-genetic already ran on the synthetic
# (continuous-only) dataset in E4 (dice_ml's genetic optimizer, its own
# defaults, NOT MOC's), giving a real, measured population-based-EA-under-
# TabPFN datapoint. Combined with PACE's own measured call counts (also
# from E4) and MOC's *published* defaults (population=20, generations=175,
# from Dandl et al.), this gives three things on one axis:
#   1. PACE       -- measured, this codebase, this run
#   2. DiCE-genetic -- measured, this codebase, this run (different EA
#      hyperparameters than MOC -- dice_ml's own, NOT MOC's -- so it's an
#      empirical anchor for "GA-style search is call-hungry", not a MOC
#      stand-in)
#   3. MOC        -- analytical projection using MOC's own published
#      hyperparameters, latency-calibrated against our *own* measured
#      per-call TabPFN cost from the genetic run above (same algorithm
#      family, same hardware, most defensible calibration available
#      without integrating MOC itself)
# ==========================================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = joblib.load("e4_tabpfn_cost.joblib")

SYN_DATASET = "mc_f80_red40_inf24_seed787359109"
MODEL_ORDER = ["LR", "RF", "XGB", "TabPFN"]

# MOC's published defaults (Dandl et al., MOCClassif R package): population
# (mu) ~20, generations ~175. Total forward evaluations ~= mu * (generations
# + 1) (initial population + one offspring batch per generation) -- lands
# at the ~3,500 figure the action list already cites.
MOC_POPULATION = 20
MOC_GENERATIONS = 175
MOC_CALLS_PER_CE = MOC_POPULATION * (MOC_GENERATIONS + 1)  # 3520

# ---------------------------------------------------------------------------
# 1. PACE -- measured, same synthetic dataset as DiCE-genetic below (an
#    apples-to-apples comparison: same data, same hardware, same run).
#    Call counts are dataset-invariant by construction (see E4), so this
#    matches the all-datasets-pooled numbers exactly on calls; only the
#    per-CE wall-clock differs slightly by dataset (via n_train size), which
#    is exactly why matching datasets here matters.
# ---------------------------------------------------------------------------
pace = df[(df["method"] == "pace") & (df["dataset"] == SYN_DATASET)]
pace_stats = pace.groupby("model").agg(
    calls_mean=("target_calls", "mean"),
    calls_min=("target_calls", "min"),
    calls_max=("target_calls", "max"),
    time_s_mean=("time_s", "mean"),
).reindex(MODEL_ORDER)

# ---------------------------------------------------------------------------
# 2. DiCE-genetic -- measured, synthetic set only (the only dataset E4 ran
#    DiCE under "genetic"; dice_ml's own defaults, not MOC's)
# ---------------------------------------------------------------------------
dice_gen = df[(df["dataset"] == SYN_DATASET) & (df["method"] == "dice")]
assert (dice_gen["ok"]).all(), "expected 100% completeness for DiCE-genetic on the synthetic set"
dice_gen_stats = dice_gen.groupby("model").agg(
    calls_mean=("target_calls", "mean"),
    rows_mean=("target_rows_evaluated", "mean"),
    time_s_mean=("time_s", "mean"),
).reindex(MODEL_ORDER)
dice_gen_stats["batch_size_mean"] = dice_gen_stats["rows_mean"] / dice_gen_stats["calls_mean"]
dice_gen_stats["time_per_call_s"] = dice_gen_stats["time_s_mean"] / dice_gen_stats["calls_mean"]

print("=== DiCE-genetic, synthetic set, measured (dice_ml's own defaults) ===")
print(dice_gen_stats)
print(f"\n(dice_ml genetic here: {dice_gen_stats['calls_mean'].iloc[0]:.0f} generations x "
      f"~{dice_gen_stats['batch_size_mean'].iloc[0]:.0f} population -- its own defaults, "
      f"deterministic across all 60 factuals/classifiers checked: "
      f"calls min={dice_gen['target_calls'].min()}, max={dice_gen['target_calls'].max()}.)")

# ---------------------------------------------------------------------------
# 3. MOC -- analytical projection, calibrated on DiCE-genetic's own measured
#    per-call TabPFN latency from this same codebase/hardware (same
#    algorithm family: population-based EA repeatedly querying the target
#    once per generation; this is the most defensible calibration available
#    short of integrating MOC itself).
# ---------------------------------------------------------------------------
moc_projection = pd.DataFrame({
    "model": MODEL_ORDER,
    "calls_per_ce": MOC_CALLS_PER_CE,
    "time_per_call_s_calibration": dice_gen_stats["time_per_call_s"].reindex(MODEL_ORDER).values,
}).set_index("model")
moc_projection["projected_time_s"] = (
    moc_projection["calls_per_ce"] * moc_projection["time_per_call_s_calibration"]
)

# ---------------------------------------------------------------------------
# Assemble the headline table
# ---------------------------------------------------------------------------
rows = []
for model_name in MODEL_ORDER:
    rows.append({
        "classifier": model_name, "method": "PACE",
        "target_calls_per_ce": pace_stats.loc[model_name, "calls_mean"],
        "forward_passes_per_ce": pace_stats.loc[model_name, "calls_mean"],
        "time_s_per_ce": pace_stats.loc[model_name, "time_s_mean"],
        "time_basis": "measured",
    })
    rows.append({
        "classifier": model_name, "method": "DiCE-genetic (dice_ml defaults)",
        "target_calls_per_ce": dice_gen_stats.loc[model_name, "calls_mean"],
        "forward_passes_per_ce": dice_gen_stats.loc[model_name, "calls_mean"],
        "time_s_per_ce": dice_gen_stats.loc[model_name, "time_s_mean"],
        "time_basis": "measured",
    })
    rows.append({
        "classifier": model_name, "method": "MOC (published defaults, projected)",
        "target_calls_per_ce": moc_projection.loc[model_name, "calls_per_ce"],
        "forward_passes_per_ce": moc_projection.loc[model_name, "calls_per_ce"],
        "time_s_per_ce": moc_projection.loc[model_name, "projected_time_s"],
        "time_basis": "projected (calls x DiCE-genetic's own measured per-call latency)",
    })

table = pd.DataFrame(rows)
table.to_csv("e1b_scaling_table.csv", index=False)
print("\n=== e1b_scaling_table.csv ===")
pd.set_option("display.width", 200)
print(table.to_string(index=False))

# Headline TabPFN numbers for the write-up
pace_tabpfn_t = pace_stats.loc["TabPFN", "time_s_mean"]
moc_tabpfn_t = moc_projection.loc["TabPFN", "projected_time_s"]
print(f"\nHeadline (TabPFN): PACE measured {pace_tabpfn_t:.2f}s/CE "
      f"({pace_stats.loc['TabPFN','calls_mean']:.0f} calls) vs. MOC projected "
      f"{moc_tabpfn_t:.1f}s/CE ({moc_tabpfn_t/60:.1f} min, {MOC_CALLS_PER_CE} calls) "
      f"-> {moc_tabpfn_t/pace_tabpfn_t:.0f}x slower, projected.")

# ---------------------------------------------------------------------------
# Bar chart: target calls per CE, PACE vs GA-style search, per classifier
# (log y-axis -- PACE/DiCE-genetic/MOC span ~2 to ~3520 calls).
# Colors: same 3 of the 8 Okabe-Ito hues used for method identity in
# e4_analysis.py's PACE/DiCE colors, plus a distinct third for the MOC
# projection (a different visual treatment -- hatched -- since it's
# projected, not measured, and this distinction matters for an honest
# reading of the chart).
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "PACE": "#0072B2",
    "DiCE-genetic (dice_ml defaults)": "#D55E00",
    "MOC (published defaults, projected)": "#949494",
}
METHOD_HATCH = {
    "PACE": None,
    "DiCE-genetic (dice_ml defaults)": None,
    "MOC (published defaults, projected)": "///",
}

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(MODEL_ORDER))
width = 0.25
methods = ["PACE", "DiCE-genetic (dice_ml defaults)", "MOC (published defaults, projected)"]
for i, method in enumerate(methods):
    vals = table[table["method"] == method].set_index("classifier").reindex(MODEL_ORDER)["target_calls_per_ce"]
    ax.bar(
        x + (i - 1) * width, vals, width=width,
        color=METHOD_COLORS[method], hatch=METHOD_HATCH[method],
        edgecolor="white" if METHOD_HATCH[method] else "none",
        label=method,
    )
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(MODEL_ORDER)
ax.set_ylabel("target-model calls per CE (log)")
ax.set_title("Target-model calls per counterfactual: PACE vs. population-based EA search")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.grid(True, axis="y", which="both", alpha=0.15)
fig.tight_layout()
fig.savefig("e1b_calls_per_ce_bar.pdf", bbox_inches="tight")
print("\nWrote e1b_scaling_table.csv and e1b_calls_per_ce_bar.pdf")
