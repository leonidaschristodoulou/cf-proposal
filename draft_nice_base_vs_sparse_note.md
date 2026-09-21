# NICE-base vs. NICE-sparse: methods/limitations note

**Grounding note:** mechanism verified directly from the installed `nice` package
source (`nice/__init__.py`, `nice/utils/optimization/heuristic.py`) and from
`output_realdata.joblib`'s actual status breakdown, not asserted from documentation.

---

We report NICE-sparse (`optimization='sparsity'`) as the sole NICE baseline on
real-world data, and both NICE variants on synthetic data. The reason is
architectural, not a matter of convenience: NICE-base's search is a single
nearest-neighbour lookup with no refinement step (`explain()` returns the nearest
training point of the desired class outright), so it has no mechanism to enforce
immutability constraints — a returned counterfactual satisfies an immutable
feature only if the nearest neighbour happens to coincide with the factual on that
feature's value, which is rare enough by chance that on datasets with an immutable
numeric *and* an immutable categorical feature simultaneously, NICE-base's
completeness collapses to 1–7% (verified: `credit-g`, `heart_disease`, `ilpd`,
`sick`; every failure is `status="violates_immutable"`, and $\ge$96\% of all
attempts on these datasets fail this way). NICE-sparse's iterative, sparsity-seeking
refinement (`best_first.optimize`, which greedily moves the fewest features
necessary toward the neighbour and stops the moment the class flips) gets
substantially better, though still incidental, protection: if a flip is achievable
without touching the immutable feature, the greedy process is likely to stop before
reaching it. This is a side effect of pursuing sparsity, not an explicit
immutability mechanism — completeness on the same four datasets rises to 60–94\%
under NICE-sparse, better but still short of the near-complete rates seen on
datasets with no immutable features at all.

Synthetic datasets carry no semantic feature identities and are generated without
any immutability specification, so this limitation cannot manifest there — both
NICE variants are directly comparable on synthetic data, and we report both.
Real-world datasets in our benchmark routinely mark at least one feature immutable
(only `breast_cancer`, `diabetes_binarized`, `australian`, and `blood_transfusion`
have none), so including NICE-base there would not be measuring the same thing as
the other baselines: its low completeness would reflect an absent constraint-handling
mechanism rather than a genuine difficulty finding counterfactuals. We therefore use
NICE-sparse as the real-world NICE baseline throughout, and note explicitly that this
is a known limitation of plain nearest-neighbour retrieval as a counterfactual
method, not of the NICE-sparse variant we do report.

**Addendum (pooled, per-classifier view — confirms the same mechanism, not a new
finding).** The completeness *figure* (`real_flip_rate.pdf`, `cfprop_plots.ipynb`
cell 11 Panel A) previously omitted `nice_base` entirely for real data — a plotting
bug (it reused the `methods` list that's deliberately `nice_base`-free for the
$\ell_0$/$\ell_2$ quality comparisons above, for the ones/completeness panel too,
where that exclusion doesn't apply — completeness is exactly the kind of uniform,
E2-mandated reporting `nice_base` *should* appear in). Fixed by giving Panel A its
own `completeness_methods = sorted(df["method"].unique())`. With `nice_base` now
visible, the pooled-per-classifier completeness (all 10 datasets, from
`real_e2_reliability.csv`) tells the identical story as the four-dataset numbers
above, just averaged more broadly: 62.5% (LR), 63.1% (RF), 63.1% (XGB), and 37.0%
(TabPFN) — TabPFN is the hardest classifier for `nice_base` specifically because
1284 of its 3067 failures are *also* `no_flip` on top of 1783 `violates_immutable`,
i.e. even where immutability isn't the blocker, TabPFN's decision boundary is
harder for a plain nearest-neighbour lookup to cross than the other three
classifiers'. This is a pooled cross-check that the mechanism described above holds
generally, not a different or new explanation.
