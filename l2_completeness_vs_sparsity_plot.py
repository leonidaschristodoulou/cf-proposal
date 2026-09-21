# ==========================================================================
# L2: completeness vs. sparsity plot across the 4 recovery cases
# (breast_cancer, australian, ilpd, heloc, all TabPFN). One point per
# (dataset, method): x = median l0, y = completeness (%).
#
# The denominator is ALL of PACE-default's original failures for that
# dataset (target_failed, e.g. 45 for ILPD, 56 for HELOC) -- the same
# denominator as the recovery-rate table -- NOT PACE-relaxed's own
# C_more_feats-recovered subset. Using the recovered subset as the
# denominator was circular: it made PACE 100% complete by construction
# (tautological, not a finding) and silently restricted every baseline's
# comparison to only the instances PACE itself already solved, hiding how
# they do on the harder residual instances PACE-relaxed doesn't recover
# either. Every method, PACE included, is now scored against the same fixed
# set of originally-hard instances.
#
# Methods with ZERO successes on a dataset's target set are plotted at
# (0, 0) with a hollow marker (no meaningful l0 exists for a method that
# never succeeded -- x=0 is a placeholder, not a measurement) rather than
# omitted, since silent omission looked like an error rather than the
# finding it is. The dominant failure status behind each such (0,0) point
# is printed at the end for the caption.
# ==========================================================================

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "/nvme/h/lchristodoulou/pace/cf-proposal/output_realdata.joblib"
# (dataset, display label, panel position in the 2x2 grid)
CASES = [
    ("breast_cancer", "Breast Cancer"),
    ("australian", "Australian"),
    ("ilpd", "ILPD"),
    ("heloc", "HELOC"),
]
MODEL = "TabPFN"

COLORS = {"nice_base": "#0072B2", "nice_spars": "#E69F00", "dice": "#D55E00", "mcce": "#CC79A7", "pace": "#009E73"}
LABELS = {"nice_base": "NICE (base)", "nice_spars": "NICE (sparse)", "dice": "DiCE", "mcce": "MCCE", "pace": "PACE (relaxed)"}
MARKERS = {"nice_base": "o", "nice_spars": "s", "dice": "D", "mcce": "^", "pace": "*"}

df_all = joblib.load(RESULTS_PATH)

fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharey=True)
axes = axes.flatten()

zero_completeness_log = []  # (dataset, method, dominant_status, n) for the caption

for panel_idx, (ax, (dataset, dataset_label)) in enumerate(zip(axes, CASES)):
    tag = f"{dataset}_{MODEL}"
    sweep_df = joblib.load(f"l2_sweep_{tag}.joblib")

    # Fixed comparison set: ALL of PACE-default's original failures for this
    # (dataset, model) -- not just the subset C_more_feats itself recovers.
    pace_default = df_all[
        (df_all["dataset"] == dataset) & (df_all["model"] == MODEL) & (df_all["method"] == "pace")
    ]
    target_failed_set = set(zip(
        pace_default.loc[pace_default["status"] != "ok", "seed"],
        pace_default.loc[pace_default["status"] != "ok", "idx"],
    ))
    n_target = len(target_failed_set)

    pace_mf = sweep_df[(sweep_df["config"] == "C_more_feats") & sweep_df["found"]].copy()
    pace_recovered_set = set(zip(pace_mf["seed"], pace_mf["idx"]))
    assert pace_recovered_set <= target_failed_set, "recovered set must be a subset of the original failures"

    other_raw = df_all[
        (df_all["dataset"] == dataset) & (df_all["model"] == MODEL)
        & (df_all["method"].isin(["nice_base", "nice_spars", "dice", "mcce"]))
    ].copy()
    on_target = other_raw[
        other_raw.apply(lambda r: (r["seed"], r["idx"]) in target_failed_set, axis=1)
    ]
    other_ok = on_target.dropna(subset=["l0", "l2"])

    points = []
    n_zero_seen = 0
    for m in ["dice", "nice_base", "nice_spars", "mcce"]:
        sub = other_ok[other_ok["method"] == m]
        n_succ = len(sub)
        completeness = 100.0 * n_succ / n_target if n_target else np.nan
        if n_succ:
            med_l0 = sub["l0"].median()
        else:
            # Placeholder -- no successful CE, no l0 to measure. Small
            # horizontal jitter per zero-completeness method in this panel
            # so multiple such methods (e.g. both NICE variants on
            # Australian) don't render exactly on top of one another.
            med_l0 = 0.4 * n_zero_seen
            n_zero_seen += 1
            m_all = on_target[on_target["method"] == m]
            dominant_status = m_all["status"].mode().iloc[0] if len(m_all) else "?"
            zero_completeness_log.append((dataset, m, dominant_status, len(m_all)))
        points.append((m, med_l0, completeness, n_succ))

    n_pace_recovered = len(pace_recovered_set)
    pace_completeness = 100.0 * n_pace_recovered / n_target if n_target else np.nan
    points.append(("pace", pace_mf["l0"].median(), pace_completeness, n_pace_recovered))

    for m, med_l0, completeness, n_succ in points:
        hollow = (n_succ == 0)
        ax.scatter(med_l0, completeness, s=140,
                   color=("white" if hollow else COLORS[m]), marker=MARKERS[m],
                   edgecolors=COLORS[m], linewidths=1.6 if hollow else 0.8,
                   alpha=0.9, zorder=5, label=LABELS[m])

    ax.set_ylim(-5, 105)
    ax.set_xlim(left=-0.5)
    if panel_idx == 0:
        # Breast Cancer (top-left): legend in the lower-left; dataset text
        # in the lower-right, same position as the other 3 panels. The
        # legend is wide (5 entries) but stays within the left ~45% of the
        # panel, well clear of the right-anchored text.
        ax.legend(fontsize=8, loc="lower left", frameon=False)
        ax.text(0.97, 0.03, f"{dataset_label}\n(n={n_target} instances)", transform=ax.transAxes,
                 fontsize=11, fontweight="bold", va="bottom", ha="right")
    else:
        # Other 3 panels: no legend (shared with panel 1); dataset info
        # where the legend used to sit.
        ax.text(0.97, 0.03, f"{dataset_label}\n(n={n_target} instances)", transform=ax.transAxes,
                 fontsize=11, fontweight="bold", va="bottom", ha="right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="both", width=0.8, labelsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    row, col = divmod(panel_idx, 2)
    if row == 0:
        # Top row: no x-tick numbers -- each column's x-scale differs from
        # its bottom-row partner anyway, and joining the rows with no
        # gap makes numbers on both rows read as one (wrong) shared axis.
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel(r"Median $\ell_0$ on this instance set", fontsize=10)
    if col == 0:
        ax.set_ylabel("Completeness (%)", fontsize=10)

fig.patch.set_facecolor("white")
plt.subplots_adjust(hspace=0.06, wspace=0.03)
plt.savefig("l2_completeness_vs_sparsity.pdf", format="pdf", dpi=300, bbox_inches="tight")
print("Saved l2_completeness_vs_sparsity.pdf")

print("\nZero-completeness cases (for the caption):")
for dataset, m, status, n in zero_completeness_log:
    print(f"  {dataset} / {LABELS[m]}: 0/{n} succeeded, dominant status = {status}")
