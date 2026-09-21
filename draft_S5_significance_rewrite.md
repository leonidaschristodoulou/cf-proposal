# S5 — rewritten significance text for Sections 4–6

**Superseded:** this text has been pasted directly into `pace.tex` (found
checked out in the repo root after all — the "isn't checked out" note below
was wrong). Kept here as the standalone real-data-only version for reference;
`pace.tex` also now covers the mock/synthetic-data addendum (30-dataset
suite, `nice_spars`-only comparable, same crossover pattern replicates
independently — see `REVISION_EVIDENCE_LOG.md` → "Workstream 4" → "Mock/synthetic-data
addendum"), which this file does not include.

**Grounding note:** every number below is read directly from `hier_stats`/`dataset_stats`
computed by `cfprop_plots.ipynb` section 3c′ (`hierarchical_l2_test`/
`combine_hierarchical_results`) and 3c (`dataset_level_wilcoxon`) against
`output_realdata.joblib`, `paired_kwargs` with `min_n=5`. See
`REVISION_EVIDENCE_LOG.md`'s "Workstream 4" entry for the full derivation, the
bug found and fixed while producing these numbers, and the script used to
regenerate them (`ground_s5.py`, not checked into the repo — a throwaway
grounding script; rerun the 3c′ cells in the notebook to reproduce).

---

## Statistical methodology (wherever Section 4 describes the testing procedure)

> To assess whether PACE's counterfactuals are closer (lower $\ell_2$) than
> each baseline's at matched sparsity, we condition on exact $\ell_0$
> agreement between the two methods' outputs for a given (dataset, classifier,
> factual, seed) instance and compare the paired $\ell_2$ values. Seeds and
> factuals drawn from the same dataset are not independent observations, so
> naively pooling all matched pairs as i.i.d. and running one signed-rank test
> per (baseline, classifier, $\ell_0$) cell overstates the effective sample
> size. We instead fit a random-intercept mixed model,
> $\Delta\ell_2 = \ell_2^{\text{PACE}} - \ell_2^{\text{baseline}} \sim 1 + (1 \mid \text{dataset})$,
> per (baseline, classifier, $\ell_0$) cell, and assess the population-level
> mean effect with a dataset-level cluster bootstrap (resampling *which* of
> the $\le 10$ datasets contribute, retaining every instance from each
> resampled dataset) rather than the mixed model's own asymptotic p-value,
> which is known to run anti-conservative with this few clusters. All
> reported p-values are Holm–Bonferroni corrected across the full family of
> (baseline $\times$ classifier $\times$ $\ell_0$) comparisons. A stricter
> sensitivity check that collapses each dataset to a single median
> $\Delta\ell_2$ first and tests across the $\le 10$ resulting numbers finds
> no cell significant after correction at this sample size (0/67) — expected,
> since it discards all within-dataset replication; we report the
> instance-level hierarchical test as primary because it is the one that
> actually has power to detect the effects below, while still respecting the
> dataset-level independence structure.

## vs. DiCE

> After correction, PACE finds significantly closer counterfactuals than DiCE
> (mean $\Delta\ell_2 < 0$, $p_{\text{adj}} < 0.05$) in 21 of 23 testable
> (classifier $\times$ $\ell_0$) cells, spanning all four classifiers and
> nearly every sparsity level tested ($\ell_0 = 1$–$7$); the two
> non-significant cells (LR, $\ell_0 \in \{3, 4\}$) have $n \le 52$ instances
> across $\le 5$ datasets and are underpowered rather than null. Effect sizes
> are substantial and consistently signed: mean $\Delta\ell_2$ ranges from
> $-0.48$ to $-6.00$ across significant cells (e.g. LR/$\ell_0=1$: mean
> $\Delta\ell_2 = -2.72$, 95% CI $[-3.66, -1.72]$, $n=1014$ across all 10
> datasets), and PACE Pareto-dominates DiCE in 92.3% of comparable matched
> pairs overall (13614/14756).

## vs. MCCE

> The comparison against MCCE is far less clear-cut: only 5 of 21 testable
> cells reach significance after correction, and the direction is mixed —
> PACE is significantly closer in 3 cells (RF/$\ell_0=5$, XGB/$\ell_0=1$,
> XGB/$\ell_0=5$) but significantly farther in 2 (RF/$\ell_0=2$,
> TabPFN/$\ell_0=2$). We attribute the weak statistical picture largely to
> MCCE's low rate of matching PACE's $\ell_0$ exactly: MCCE selects the
> lowest-$\ell_0$ candidate among its samples rather than targeting a
> specific sparsity, so most (classifier, $\ell_0$) cells contain matched
> instances from only 1–3 of the 10 datasets (13 of 21 testable cells have
> $k_{\text{datasets}} \le 2$) — underpowered for the conditional test, and
> the likely reason the unconditional Pareto dominance rate (PACE dominates
> 96.6% of comparable pairs, 15751/16306) reads so much more favorably: the
> two statistics answer different questions (dominance uses every pair
> regardless of matched sparsity; the conditional test only the rare ones
> where MCCE happens to match PACE's $\ell_0$). We report the matched-$\ell_0$
> comparison as inconclusive rather than a win, consistent with S4.

## vs. NICE-sparse

> The comparison against NICE-sparse shows a genuine, sparsity-dependent
> crossover that a single pooled claim would obscure: PACE is significantly
> closer at $\ell_0=1$ across all four classifiers (mean $\Delta\ell_2$ from
> $-0.42$ to $-0.58$, all $p_{\text{adj}} < 0.001$, $n \ge 700$ instances
> across all 10 datasets in every case), but the effect reverses at higher
> sparsity: NICE-sparse becomes significantly closer at $\ell_0 \ge 6$ for RF
> ($\ell_0=6,7,8$: mean $\Delta\ell_2 = +1.34, +1.50, +1.75$, all
> $p_{\text{adj}} < 0.001$), XGB ($\ell_0=6,7$: $+1.27, +1.31$), and TabPFN
> ($\ell_0=6,7$: $+2.60, +1.79$). 13 of 32 testable cells reach significance
> overall. We report this as a genuine trade-off, not an unqualified win:
> PACE is the stronger choice in the low-sparsity regime that matters most
> for actionable, interpretable counterfactuals, while NICE-sparse's
> iterative nearest-neighbour refinement finds closer points once more
> features are allowed to change. Pooled Pareto dominance (PACE dominates
> 75.8% of comparable pairs, 8506/11228) is consistent with this picture — the
> weakest dominance rate of the three baselines, reflecting the high-$\ell_0$
> cells where NICE-sparse wins.

## Overall framing (Section 6 / discussion)

> Statistical significance after correction is not uniform across baselines,
> and we report it as such rather than defaulting to a single omnibus claim:
> strong and consistent against DiCE (21/23 cells), a genuine
> sparsity-dependent trade-off against NICE-sparse (13/32, direction flips
> around $\ell_0=6$), and inconclusive against MCCE (5/21, undermined by
> MCCE's low rate of matching PACE's $\ell_0$ exactly). Effect sizes and
> Pareto dominance rates — descriptive statistics that do not depend on the
> significance-testing machinery above — tell a consistent story: PACE
> dominates 92–97% of comparable pairs against DiCE and MCCE, and 76% against
> NICE-sparse, with the NICE-sparse gap concentrated at high sparsity.

---

**Not yet done:** paste into the actual manuscript sections and replace the
placeholder cross-references; confirm classifier-name capitalization and
$\ell_0$/$\ell_2$ macro usage match the paper's existing LaTeX conventions
(this draft uses the same math-mode style as the rest of this repo's drafted
text, e.g. `draft_nice_base_vs_sparse_note.md`, `A3` in
`REVISION_EVIDENCE_LOG.md`, but wasn't checked against the manuscript's own
macros since it isn't checked out here).
