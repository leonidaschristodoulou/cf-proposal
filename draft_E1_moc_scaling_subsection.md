# E1 — MOC positioning + scaling-analysis subsection (Section 4)

**SUPERSEDED for the manuscript text.** Once `pace.tex` was shared, this content
was pasted directly into its `\subsection{Model-call accounting and scaling}`
(`sec:calls`), replacing that section's `\revitem{E4}`/`\revitem{E1, E1b}` notes —
using the manuscript's real bib key (`dandl2020multi`, not the `dandl2020moc`
placeholder guessed below) and real section labels. See
`REVISION_EVIDENCE_LOG.md`'s E1b entry for the exact edit. This file is kept for
the condensed **rebuttal-letter paragraph** below, which hasn't been pasted
anywhere yet (no rebuttal-letter file exists in this repo) — everything else below
is superseded.

**Grounding note:** every number below is read directly from `e1b_scaling_table.csv`
(`e1b_scaling_analysis.py`, itself built on `e4_tabpfn_cost.joblib` — see
`REVISION_EVIDENCE_LOG.md`'s "E4" and "E1b" entries for full derivation and the
implementation choices behind them). This is a draft for pasting into the
manuscript, not final prose — the paper source isn't checked out in this repo, so
citation keys, section/label references, and the exact related-work wording it
should sit next to are placeholders. **Please share the current Related Work
section and Section 4's baselines paragraph if you'd like this tightened to match
exactly** — in particular I don't know: the actual bib key for Dandl et al. 2020,
whether R1w (MOC's related-work positioning) has already been drafted elsewhere,
and where in Section 4 the "baselines" are currently introduced.

Two versions below: the manuscript subsection (Section 4, paste-ready modulo the
placeholders above), and a condensed rebuttal-letter paragraph for the R2/R3
response addressing "I miss a comparison" / "could be compared."

---

## Manuscript text (Section 4 — Experimental Setup)

> **On multi-objective / evolutionary comparison (MOC).** We do not include a
> direct empirical comparison against MOC~\citep{dandl2020moc}, the representative
> Pareto-based multi-objective/evolutionary counterfactual method discussed in
> Section~\ref{sec:related_work}, for two reasons. First, availability: MOC's
> reference implementation is the R package `counterfactuals`/`MOCClassif`;
> integrating it faithfully into our Python/TabPFN pipeline — reproducing its
> mixed-integer evolutionary search (MIES) and Gower-distance handling of mixed
> feature types — is a substantial engineering undertaking whose payoff is
> undercut by the second reason. Second, and more fundamental: MOC is a
> population-based evolutionary search that queries the target predictor once per
> individual per generation, a cost structure fundamentally mismatched to
> expensive, in-context predictors such as TabPFN (Section~\ref{sec:efficiency}).
> Rather than assert this mismatch, we quantify it below.
>
> We are not, however, avoiding population-based evolutionary search as a class.
> DiCE's genetic optimizer (`dice-genetic`) is a baseline throughout this paper on
> the continuous-only synthetic datasets (Section~\ref{sec:exp_setup}), where its
> handling of the search space is uncontroversial (see Section~\ref{sec:dice_split}
> for why genetic is used there and random on the mixed real-world sets).
> `dice-genetic` gives us a real, measured instance of population-based
> evolutionary counterfactual search, under our own hardware and protocol, which we
> use to ground the scaling argument below in place of an unintegrated MOC run.
>
> **Quantified scaling argument.** Table~\ref{tab:moc_scaling} reports, per
> classifier, target-model calls per counterfactual (CE) and the resulting
> wall-clock cost, for three points on the same axis: PACE (measured), `dice-genetic`
> (measured — its own default hyperparameters, not MOC's: 16 generations of a
> $\approx$51-individual population, deterministic across every instance we
> checked), and MOC (**projected**, using its published defaults — population
> $\approx 20$, generations $\approx 175$~\citep{dandl2020moc}, giving
> $20 \times 176 = 3{,}520$ target queries per CE — latency-calibrated against
> `dice-genetic`'s own measured per-call cost under TabPFN on this same hardware,
> the most direct calibration available short of integrating MOC itself). Under
> TabPFN, PACE requires 2 target-model calls per CE (1.07s, measured); the same
> projection for MOC is 3,520 calls, $\approx$18.9 minutes per CE — roughly three
> orders of magnitude slower ($\approx 1{,}064\times$).
>
> The gap is not that MOC's individual queries are expensive: `dice-genetic`'s own
> measured per-call cost under TabPFN (0.32s, at a population batch of $\approx$51)
> is the same order of magnitude as PACE's own per-call cost ($\approx$0.53s,
> averaged over PACE's two calls, one of which scores a batch of $\approx$2,000
> candidates in a single pass). The gap is architectural: a population-based search
> must issue one such query *per generation, sequentially* — each generation's
> population depends on the previous generation's fitness evaluation — so its total
> cost scales with population $\times$ generations, while PACE amortizes an entire
> candidate pool into a single batched call. This is the same batching mechanism
> that explains PACE's stable runtimes under TabPFN elsewhere in the paper
> (Section~\ref{sec:efficiency}); the analysis here is additional evidence for that
> mechanism, not a separate claim.
>
> We report this as an analytical, call-count argument calibrated on our own
> measurements, not a head-to-head benchmark result, and consider it a more
> informative use of the available effort than a first, unoptimized MOC
> integration would have been.

### Table (LaTeX, `booktabs`, matches this repo's existing table style e.g. `real_accuracy_auc_table.tex`)

```latex
\begin{table}[t]
\centering
\caption{Target-model calls and wall-clock cost per counterfactual (CE):
PACE (measured) vs.\ population-based evolutionary search. \textsc{DiCE-gen}
is measured under \texttt{dice\_ml}'s own default hyperparameters (not MOC's).
\textsc{MOC} rows are \emph{projected} from its published defaults
(population $\approx 20$, generations $\approx 175$), latency-calibrated
against \textsc{DiCE-gen}'s own measured per-call cost on the same hardware
--- marked \emph{proj.} throughout, never a measured result.}
\label{tab:moc_scaling}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{2}{c}{\textbf{LR}} & \multicolumn{2}{c}{\textbf{XGB}} & \multicolumn{2}{c}{\textbf{RF}} & \multicolumn{2}{c}{\textbf{TabPFN}} \\
\cmidrule(lr){2--3} \cmidrule(lr){4--5} \cmidrule(lr){6--7} \cmidrule(lr){8--9}
 & \textsc{calls} & \textsc{time (s)} & \textsc{calls} & \textsc{time (s)} & \textsc{calls} & \textsc{time (s)} & \textsc{calls} & \textsc{time (s)} \\
\midrule
PACE (measured)              & 3     & 0.17  & 2     & 0.17  & 2     & 0.25  & 2     & 1.07 \\
\textsc{DiCE-gen} (measured) & 16    & 0.30  & 16    & 0.35  & 16    & 0.93  & 16    & 5.15 \\
MOC (proj.)                  & 3{,}520 & 65.1  & 3{,}520 & 76.1  & 3{,}520 & 204.7 & 3{,}520 & 1{,}133.6 \\
\bottomrule
\end{tabular}
\end{table}
```

(Numbers pulled directly from `e1b_scaling_table.csv`; column order LR/XGB/RF/TabPFN
matches `real_accuracy_auc_table.tex`'s existing convention. PACE's LR column is 3,
not 2 — a real, documented edge case: PACE's guidance model is always LR, so when
LR is *also* the target, the guidance-side and target-side calls coincide and can't
be told apart; see E4's evidence-log entry.)

**Figure alternative:** `e1b_calls_per_ce_bar.pdf` (already generated) — log-scale
grouped bar chart of the same calls-per-CE numbers, MOC's bars hatched to mark them
as projected rather than measured. Spec asked for "table or bar chart"; both exist,
pick whichever fits the page better next to the prose above.

---

## Condensed rebuttal-letter paragraph (R2 "I miss a comparison" / editor "could be compared")

> We considered benchmarking against MOC~\citep{dandl2020moc}, the natural
> multi-objective/evolutionary comparator, but decided against it: MOC's reference
> implementation (R, `counterfactuals`/`MOCClassif`) has no clean Python/TabPFN
> integration, and a faithful port is disproportionate effort for a method our
> results already suggest is architecturally mismatched to expensive predictors.
> We are not avoiding evolutionary methods as a class — DiCE's genetic optimizer is
> already a baseline in our benchmark on the continuous-only datasets — and we use
> its measured, real-hardware call counts to ground a quantitative scaling argument
> (new Section~\ref{sec:moc_scaling}/Table~\ref{tab:moc_scaling}) rather than assert
> the mismatch: projected under TabPFN, MOC's published defaults require
> $\approx$1,064$\times$ more wall-clock than PACE per counterfactual, because a
> population-based search must query the target once per generation sequentially,
> while PACE batches its entire candidate pool into a single pass. We believe this
> quantified projection, calibrated on our own measurements, is more informative
> than a first, unoptimized MOC integration would have been, and we have made the
> reasoning and its limits explicit in the manuscript rather than declining
> silently.

---

**Not yet done:** paste into the manuscript at the right point in Section 4 (need
the actual "Experimental Setup" text to know exactly where this subsection should
sit relative to the baselines paragraph); confirm the bib key for Dandl et al. 2020
(used `dandl2020moc` as a placeholder — action list's R1w item cites it as *PPSN
XVI, DOI 10.1007/978-3-030-58112-1_31*); confirm `\label{sec:related_work}` /
`\label{sec:efficiency}` / `\label{sec:exp_setup}` / `\label{sec:dice_split}` match
the paper's real labels; decide table vs. figure (or both) for the calls-per-CE
data; and check this doesn't duplicate content once R1w (MOC's Section 2
positioning) is actually drafted — R1w and this subsection are meant to be
complementary (R1w = what MOC *is* and how it differs methodologically; this
subsection = why it isn't *run* here, quantified).
