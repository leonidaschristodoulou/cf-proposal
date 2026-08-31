# PACE — Revision Action List

**Manuscript:** *PACE: Proposal Assembly for Counterfactual Explanations*
**Decision:** Major revision (presentation/methodology + experiments)
**Repos:** the PACE **method** (Stage A/B/C, `core.py` — importance, generation, selection) lives in `github.com/leonidaschristodoulou/pace`; the **experiments** (benchmark harness, `run_benchmark`, dataset prep, DiCE/NICE/MCCE wiring, figure/table generation, the notebook) live in `github.com/leonidaschristodoulou/cf-proposal`. Most [CODE] items here touch **cf-proposal** (protocol, instrumentation, ablations, figures, the DiCE-TabPFN fix); items that change the method itself — how importance/candidates/selection work — touch **pace/`core.py`**.

## For Claude Code

This file is meant to be read directly by Claude Code. To use it:

- **Place it inside the project.** Claude Code only sees files in the folder it's launched from, so keep this file in the repo you're working in. For most [CODE] items that's **cf-proposal** (the experiments); for method changes it's **pace**. Putting a copy in both (or in a shared `docs/`/`revision/` folder) is fine.
- **Know which repo an item touches.** Experiment/analysis items — the common protocol (E2), TabPFN instrumentation (E4), ablations (A1–A2), figures (F1–F4), the DiCE-TabPFN correctness fix (E5b), the DiCE-genetic reruns (E5), the trade-off/recovery reruns (L1–L2) — live in **cf-proposal** (`run_benchmark`, dataset prep, the notebook, figure code). Items that change the method itself — importance computation, candidate generation, selection, new hyperparameters exposed for ablation — live in **pace** (`core.py`: `stageA_build`, `_perm_importance_proba_drop`, `stageB_generate`, `stageC_select_best`). A few items span both (e.g. an ablation may add a knob in `pace` and sweep it from `cf-proposal`).
- **Ground the plan in the real repo first.** These *Spec* blocks were written from the paper and the two code files you shared, so they name real functions (`run_benchmark`, `stageC_select_best`, etc.) — but paths and surrounding structure still need locating. Good first prompt, run in each repo: *"Read this action list, then explore this repo and tell me which files each [CODE] item would touch here."* Do this before implementing anything.
- **Work one item at a time.** Prompt by ID, e.g. *"Implement E4 from PACE_revision_action_list.md"* or *"Do S1 and S2."* Each [CODE] Spec is self-contained enough to act on. One item per session/commit is more reliable than a whole workstream at once.
- **[CODE]** items are for Claude Code (calculations, scripts, figures) — mostly in **cf-proposal**, method changes in **pace**. **[TEXT]** items are manuscript writing; Claude Code can help with these too if the paper source (e.g. LaTeX) is checked out alongside, but they aren't computational and don't belong to either code repo.
- **Check intermediate output**, especially the statistics rework (S1–S5): verify the dataset-level aggregation matches your data structure before it propagates into every table and claim.

## How to use this list

- **[CODE]** = run in Claude Code against the relevant repo (cf-proposal for experiments, pace for method changes). Each has a *Spec* block precise enough to paste in.
- **[TEXT]** = manuscript writing/editing. No computation needed.
- **[FIG]** = figure regeneration (mix of code + editing).
- **Refs:** R1/R2/R3 = the three reviewers; ED = editor. Items flagged by several people are the highest priority.
- Suggested execution order is in the last section, plus a comment→section map for your rebuttal letter.

A recurring theme cuts across almost every reviewer: **the method is presented as a recipe, not a formalism, and several symbols are used before they are defined.** Fixing that (Workstream 1) resolves a large share of the individual comments.

---

## Workstream 1 — Formalise the method (Section 3)

Reviewers R1, R2, R3 and ED all flag Section 3. R3 is explicit: define the problem *before* the algorithm and give each stage its own subsection.

**M1 [TEXT] — Add a formal problem statement before Algorithm 1.** (R3, ED)
Introduce, in this order, with notation fixed up front:
- factual instance x_f, predictor f: R^d → [0,1], threshold τ, desired class y_des;
- **feature units** U (numeric singletons + one-hot groups), mutable/immutable sets U_mut, U_imm;
- **proposal space** and the three **proposal types** (anchor / boundary / noise) as maps from (x_f, reference pool) to a candidate;
- **candidate set** C, **feasibility constraints** (true class flip, valid one-hot, range/immutability), **scoring/filter** rule, and the **lexicographic selection** rule (ℓ0 → ℓ2 → |f(x)−τ|).
Only after these are defined should the pseudocode appear.

**M2 [TEXT] — Introduce "proposal-based counterfactual generation" as a modest, explicit definition, not a grand reframing.** (R3)
State it as *a* proposed formulation. Remove/soften "revisiting the foundations of CE generation" (p.2) and the abstract's "replaces continuous optimization" framing so it reads as a candidate-generation-plus-selection scheme, not a new paradigm.

**M3 [TEXT] — Rewrite Algorithm 1 into per-stage subsections with line numbers.** (R2, R3)
- 3.1 Feature-importance estimation (Stage A) — 3.2 Boundary/anchor pools — 3.3 Candidate generation (anchor / boundary / noise) — 3.4 Range constraints — 3.5 Dedup, filter, lexicographic selection.
- Add **line numbers** to the algorithm (R2 minor).
- Each symbol used in the algorithm gets defined in the matching subsection prose *before* it appears in pseudocode.

**M4 [TEXT] — Define every symbol currently used undefined.** (R1, R3)
Explicit gaps found in the current text:
- **π_a, π_b, π_n** — appear on p.6 and in Algorithm 1 but are only given values in Table 1. Define in prose at first use.
- **α, [α_min, α_max]** — the interpolation weight. Table 1 gives `Unif[0.5,1.0]` but the algorithm uses [α_min,α_max] and the prose never says what α *is*. Define "interpolation" and "constrained interpolation" explicitly (R1 asks how constrained interpolation is implemented; R3 notes "interpolation" is never defined).
- **γ** — the importance-sharpening exponent w(u) ∝ I(u)^γ appears in Algorithm 1 but has no value in Table 1 and no definition. Add both.
- **c vs σ_base** — Table 1 lists σ_base = 0.25; Algorithm 1's σ formula uses `c`. Reconcile these (state c = σ_base, or rename) so the noise scale is unambiguous.
- Also define at first use: feature units, one-hot groups, immutable sets, proposal pools, importance weights, clipping to training quantiles, τ, k_A, k_B, R, M, s, p_ch.

**M5 [TEXT] — State the design rationale / motivation for each of the three mechanisms.** (R1, R3, ED)
For anchor, boundary, and guided-noise, add one short paragraph each answering: *what regime is it for, and what does removing it cost?* (e.g., anchors = "already on the desired side, cheap and sparse"; boundary = "for diffuse/complex boundaries where few training points already cross"; noise = "reaches feasible regions no training point covers"). Your own Fig. 9 data supports this — cite it (anchors dominate generally; boundary dominates on Blood Transfusion/HELOC; noise non-negligible for XGB). This is the narrative the ablation (A1) will quantify.

**M6 [TEXT] — Clarify the guidance/surrogate model. (VERIFIED from code.)** (R1, R3 — asked twice)
The code (`core.py` + benchmark) confirms a specific two-model architecture — state it plainly in Section 3, because R3 twice said it wasn't clear:
- **PACE uses two models.** A **guidance** model drives Stage A (permutation importance, anchor/boundary pools, `p_train`) and Stage B (candidate generation); the **target** model is used only for the factual's class `p_f` and Stage C selection. In the benchmark, guidance = **logistic regression for every target** (`pp_guidance = proba_fn(models["LR"])`), so importance/generation always run on cheap LR regardless of the true model.
- Feature importance is **global permutation importance** (mean |Δ predicted-proba| after permuting each feature unit; `_perm_importance_proba_drop`), computed on the **LR guidance model** — so it is **global**, **not SHAP**, **not post-hoc on the target** (answers R1's "local or global? SHAP or model access?").
- PACE still **validates and selects against the true target** (Stage C scores candidates on the target), so PACE is *not* "explaining a surrogate" — the returned CFs are genuine target-model counterfactuals.
- **Answer R3's repeated question explicitly: only PACE uses the surrogate; DiCE, NICE, and MCCE all query the target model directly.** This asymmetry is real and currently unstated. Frame it as the *design*: PACE deliberately confines expensive-model calls to a single batched selection step by steering generation with a cheap surrogate (this is the efficiency mechanism, quantified in E4). Note the honest flip side — when the surrogate is a poor guide to the target (e.g. TabPFN's irregular surface, the Breast-Cancer case), PACE's guidance degrades, which plausibly connects to the Section 5 incompleteness.

**M7 [TEXT] — Confirm and document multiple-CE support.** (R3)
The algorithm returns a single argmin, but p.6 lists "number of CE to be returned" as a control. State that PACE returns the top-k feasible candidates under the lexicographic order, and either show a small k>1 example or say it's straightforward and defer. (Optional [CODE] to produce a k=3 example for one instance.)

**M8 [FIG] — Add a flow diagram of the pipeline.** (R2)
A single schematic: inputs → Stage A (importance + pools) → Stage B (three generators) → range filter → dedup/score/lexicographic select → CE/∅. This directly answers R2's readability request and anchors Section 3.

---

## Workstream 2 — Related work & references (Section 2)

**R1w [TEXT] — Add and position multi-objective / evolutionary CE.** (R3, R2, ED)
- Discuss **MOC (Dandl et al. 2020, PPSN XVI, DOI 10.1007/978-3-030-58112-1_31)** as the representative multi-objective/evolutionary CE method. Contrast PACE's lexicographic selection with MOC's Pareto-based multi-objective search (you avoid scalar weights *and* avoid returning a whole Pareto set; MOC returns a nondominated set the user must then choose from).
- **Use MOC to make the DiCE point precise (see E5).** MOC is a genetic method *designed for mixed data*: it replaces plain NSGA-II with **mixed integer evolutionary strategies (MIES)** to search the joint discrete–continuous space and uses **Gower's distance** for mixed feature types. This is exactly the machinery DiCE-genetic lacks. So the accurate framing is: evolutionary CF *as a class* handles mixed data (MOC proves it), but DiCE's genetic backend does not do so robustly — which is why the right evolutionary comparator on the real-world sets is MOC, not DiCE-genetic.
- **Also cite the lexicographic evolutionary CF method** (arXiv 2502.10418, 2025), which benchmarks against MOC and uses *lexicographic* optimization — the closest existing method to PACE's selection philosophy. Read it to sharpen R3w (how PACE differs from an evolutionary lexicographic search: PACE is generate-then-select, not iterative evolutionary optimisation) and to avoid being blindsided if a reviewer raises it.
This pairs with the new baseline comparison E1.

**R2w [TEXT] — Add the feature-importance-guided and manifold-constrained references.** (R3)
- **Kommiya Mothilal et al. 2021** (*Towards Unifying Feature Attribution and Counterfactual Explanations*, AIES 2021, DOI 10.1145/3461702.3462597) — R3 frames this as "similar" to your importance-guided generation, but the direction is **opposite**: that paper derives feature attributions *from* counterfactuals (and uses necessity/sufficiency to *evaluate* attributions), whereas PACE uses a global importance ranking *to guide* candidate generation. Cite it (disarms the "missed related work" complaint) but state the direction explicitly, e.g. "they derive attributions from counterfactuals; PACE inverts this, using importance to guide generation." It is **not** a CF-generation baseline and should not be treated as a near-duplicate of guided noise.
- **Tsiourvas et al.** and **Marango et al.** (constraints on the data manifold via optimization with constraint learning) — position PACE's *feasibility-as-input* approach against their *constraint-learning* approach.

**R3w [TEXT] — Sharpen the taxonomy and defend (or drop) "proposal-based" as a category.** (R3)
Add explicit one-line contrasts against sampling-based, retrieval/NN-based (NICE), generative (MCCE), and heuristic families, stating the technical difference from prior candidate-generation. If you keep "proposal-based" as a new label, justify why existing terms are insufficient; otherwise present it as a hybrid of retrieval + guided perturbation + lexicographic selection. **In particular, distinguish PACE from the lexicographic evolutionary CF method (arXiv 2502.10418):** both use lexicographic selection, but PACE is *generate-then-select* (one-shot candidate assembly + batch scoring), not an *iterative evolutionary optimisation loop* — which is precisely what makes PACE cheap under expensive predictors like TabPFN.

**R4w [TEXT] — Reference hygiene.** (R2 minor)
Ensure every reference has a DOI **or** a working link. [CODE] optional: script a check of your `.bib` for entries missing both `doi` and `url`/`eprint` and print the offenders.

*Spec (optional bib check):* parse the `.bib`, list keys where neither `doi` nor `url`/`eprint`/`arxiv` is present.

---

## Workstream 3 — Experimental protocol (Section 4)

**E1 [TEXT] — Position MOC in related work, and justify not benchmarking it. DECISION: not running it.** (R2, R3, ED)
Parse of the actual asks: **R3 and the editor require MOC/multi-objective methods to be *discussed as related work* (hard, cheap — done in R1w/R2w).** The *empirical comparison* is only a **suggestion** — R2 says "I miss a comparison… other than that the evaluation is sufficient"; the editor says it "could be compared." "Could…should also be discussed" is permissive on the comparison, mandatory on the discussion. So declining the benchmark is defensible **provided the discussion is thorough and the scaling argument is quantified** (E1b).
Chosen path and rationale to state in the response letter:
- Off-the-shelf MOC availability is messy for a Python/TabPFN pipeline (reference impl is R `counterfactuals`/`MOCClassif`; no clean drop-in), so a faithful, fair integration is disproportionate effort for a method the paper already argues is architecturally mismatched to expensive predictors.
- **You are not avoiding evolutionary methods:** you already run **DiCE with the genetic optimizer on the numerical (synthetic) datasets** — surface this explicitly in Section 4.1 as evidence that an evolutionary/GA comparator *is* in the benchmark where it is appropriate (continuous-only search space). This directly softens R2's "I miss a comparison."
- Replace the missing MOC run with the **scaling analysis in E1b**, which turns "it won't scale" from an assertion into a quantified argument.
*No code for this item* beyond making the existing DiCE-genetic (synthetic) results visible; the substance is text (R1w/R2w) + E1b.

**E1b [CODE/TEXT] — Scaling analysis: quantify why population-based evolutionary CF is mismatched to expensive predictors.** (replaces the MOC benchmark; supports R2, ED)
This is the evidence that lets you decline the MOC run without hand-waving. The goal: show, in target-model *call counts* (not just wall-clock, which is hardware-dependent), that a population-based evolutionary search is architecturally expensive under in-context predictors like TabPFN, whereas PACE's cost is a single batched pass.
*Spec (CODE):*
- **Call-count model.** For a population-based EA (NSGA-II/MIES as in MOC): forward evaluations per instance ≈ `population_size × n_generations` (plus initialisation), each a separate target-model query. Using MOC's published defaults (population ≈ 20, generations ≈ 175) that is ≈ 3,500 target calls **per counterfactual**. For PACE: `M` candidates (default 2,000) scored in `⌈M / batch_size⌉` **batched** forward passes — reuse the actual counts from E4. Tabulate both as *target-model evaluations per CE* and *number of forward passes per CE*.
- **Latency projection.** Multiply the call counts by your measured TabPFN per-call (or per-batch) latency from E4 to project per-CE wall-clock for the EA under TabPFN, and contrast with PACE's measured value. Make explicit that batching is the lever: PACE amortises 2,000 candidates into a handful of passes; an iterative EA cannot, because each generation depends on the last.
- **Optional empirical anchor (cheap, no MOC integration needed):** you already run **DiCE-genetic on the synthetic sets** — extract its target-model call counts / runtime there and use them as a real datapoint for "GA-style search is call-hungry," then argue the count only grows under TabPFN. This grounds the projection in your own measured numbers without integrating MOC.
*Spec (TEXT), Discussion / Section 4:*
- Add a short paragraph (and ideally one small table or bar chart: *target calls per CE*, PACE vs GA-style search, per classifier) making the scaling argument quantitatively.
- State the honest scope: this is an *analytical/call-count* argument, not a head-to-head MOC benchmark, and say why (implementation availability + architectural mismatch). Pre-empting the objection in the text is what makes declining the run defensible.
- Tie back to the paper's thesis: this is the mechanism behind PACE's stable TabPFN runtimes, so the scaling analysis does double duty as support for the core claim.

**E2 [CODE] — Uniform completeness & reliability reporting for all methods.** (R3, ED)
**Why the reviewer flags this (the asymmetry in the current paper).** You already have a rigorous reliability treatment — but *only for PACE*: the `no_flip` (691, 3.5%) / `no_output` (520, 2.6%) taxonomy, Table 4's per-dataset/classifier failure breakdown, Appendix B. The baselines get none of it — just scattered prose ("NICE-sparse … absent from those comparisons," "DiCE is more likely to fail") with **no counts and dropped cells**. Two distinct problems result:
1. **Asymmetric granularity (labels):** the failure vocabulary built for PACE is never applied to DiCE/NICE/MCCE, so a reader sees *that* a baseline is missing but not *why* (timeout? no output? malformed CF? not run?).
2. **Non-common denominator (deeper, not just labels):** when a baseline is "absent" because it didn't terminate, those instances appear to be *dropped* from its completeness rather than counted against it. That scores baselines on an easier subset while PACE is scored over everything including its own failures — so the completeness numbers aren't comparable. This is the "single common denominator" in the fuller comment.
You've done the hard conceptual work already; this is **extending your existing PACE taxonomy to the baselines**, not inventing anything.
*Spec:*
- Define one outcome status per (method, dataset, classifier, seed, factual) from a fixed vocabulary — e.g. `{success, no_flip, no_output, timeout, invalid, not_run}` — reusing the exact definitions you already use for PACE. `no_flip` = returned a point that didn't cross the boundary; `no_output` = returned nothing; `timeout` = exceeded the common wall-clock cap (E3); `invalid` = malformed CF (e.g. broken one-hot); `not_run` = genuinely not attempted, **with a stated reason**.
- **Assign every attempted query exactly one status — nothing gets silently dropped.** In particular, timeouts / no-output / previously-"absent" cells must be **counted in the denominator as failures**, not removed.
- Report **completeness = success / attempted over the common denominator, computed identically for every method** — including PACE. (PACE's own `no_flip`/`no_output` must sit in the *same* table under the *same* schema; do not exempt it, or a reviewer will notice instantly and it costs more credibility than the original ambiguity.)
- Produce a **"reliability" table**: the full status breakdown per method × classifier (the baseline analogue of your Table 4).
- Distinguish `not_run` (with reason) from genuine failure — don't code a config you chose not to run as either a success or a failure.
- Output a tidy CSV that every figure reads from, so completeness/reliability are guaranteed consistent across the paper.
*Cross-refs:* the common budget that makes `timeout` well-defined is the wall-clock cap in E3; the cost asymmetry that *explains* differing outcomes is the call-count table in E4. Report those alongside so differing completeness is shown to be a *result*, not an uncontrolled confound.

**E3 [CODE/TEXT] — Common budget: wall-clock cap, not an identical algorithmic budget.** (R3)
Don't over-promise here. A single *identical algorithmic* budget is not well-defined across these methods — PACE's budget is a candidate count M, DiCE-genetic's is population × generations, NICE has no comparable iteration knob, MCCE samples-then-filters. Forcing one common knob would be misleading, not rigorous. You can decline that on principled grounds *in the text*.
*Spec (CODE):*
- Impose the one budget that *is* common and meaningful: an **identical wall-clock timeout applied to every method**, on fixed hardware. Anything exceeding it is `timeout` under E2. State the limit and hardware so it's reproducible and clearly not tuned to disadvantage baselines.
- Report **target-model call counts** per method (E4) alongside, since that is the hardware-independent cost measure and explains *why* budgets differ (DiCE issues thousands of sequential calls; PACE a handful of batched ones).
*Spec (TEXT):* one or two sentences: "a single identical algorithmic budget is ill-defined across methods with heterogeneous search procedures; we instead impose a common wall-clock budget and report per-method model-call counts, making the cost asymmetry explicit rather than hidden." This converts the reviewer's budget objection into a demonstration of the paper's motivation (existing methods are brittle/expensive under expensive predictors), rather than a comparison you can't run. Keep the honest caveat (already in your text) that wall-clock depends on available compute.

**E4 [CODE] — Instrument TabPFN cost to demonstrate (not assert) the efficiency mechanism. (Mechanism VERIFIED from code.)** (R3)
The manuscript asserts batched scoring explains the speedup; `core.py` confirms *exactly* how, so you can now demonstrate it. **Verified per-factual target-model accounting for PACE:** `p_f` = 1 singleton call; Stage C = **1 batched `predict_proba` over the whole (deduped) candidate pool** (`stageC_select_best` scores `C` in a single call); Stage A/B run on the LR guidance model, so cache building makes **zero** target calls and happens once per model. Net: **2 target-model invocations per factual, one of them a single batched pass over ~n_candidates rows.** Baselines call the target iteratively throughout their search.
*Spec (CODE):*
- Wrap the target's `predict`/`predict_proba` (and TabPFN's forward) in a counter/decorator. Log, per method × dataset × classifier: **target-model row-evaluations**, **number of forward passes/batches**, candidate count (before/after dedup), batch size, and (PACE only) the one-off LR guidance fit + Stage-A importance cost.
- Tabulate **target calls per CE** across methods (`tabpfn_cost.csv`). Expected contrast: PACE ≈ 2 passes/CE; iterative baselines = many.
- **Add the demonstration plot (this is what answers "rigorously demonstrate"):** runtime vs target-call-count, one panel per classifier, all methods. Show the points fall on a rising line and PACE sits at the cheap end — the *correlation* is the proof that runtime is governed by call count, not a coincidence. Tabulating counts alone is necessary but not sufficient; the plot is the causal argument.
*Honesty check:* report whether the LR guidance/importance cost is genuinely negligible next to TabPFN calls (it should be — it's a one-off LR fit + permutation importance on LR). If for some cheap target (LR/RF) the surrogate overhead is a non-trivial share, say so; the efficiency claim is specifically about *expensive* predictors, and being precise about where it does/doesn't dominate strengthens it. Don't go in committed to proving the existing prose — if the numbers say the mechanism is "cheap-surrogate guidance + batched target scoring," state both, not batching alone.
*Timer boundary (VERIFIED — state this in the paper):* Stage A (importance + anchor/boundary pools + `p_train`) is built **once per (model, dataset, seed) outside the timed region** (`guided_cf_build_cache` runs before the factual loop; the per-CE timer wraps only `guided_cf_one_repeated` = Stage B + Stage C + the `p_f` call). So reported per-CE runtimes **exclude Stage A** — this is the intended offline/amortized design, and it is handled **symmetrically**: NICE/MCCE/DiCE explainer construction is likewise excluded from their timers. Action: (i) state explicitly in the experimental setup what the timer includes/excludes; (ii) report the one-time Stage A cost and the amortization ("cost X, amortized across N factuals, mirroring the one-time explainer builds excluded for baselines") so the amortization reads as a disclosed design advantage, not a hidden subsidy. Note Stage A is LR-only, so it makes zero target calls and does not affect E4's call-count argument either way.

**E5 [CODE/TEXT] — Replace the DiCE genetic-vs-random justification with evidence, and neutralise the fairness objection.** (R3)
**The real vulnerability (from the paper's own numbers).** The manuscript never compares the two DiCE optimizers head-to-head: genetic is used *only* on synthetic (continuous) data, random *only* on real-world (mixed) data — different datasets, so no controlled comparison exists. Completeness (approx., from Fig 2 / Fig 5 left panels — **pull exact values from your CSVs before quoting**): DiCE-genetic on synthetic ≈ 100% for LR/RF/XGB, ≈ 65% on TabPFN; DiCE-random on real-world ≈ 95/88/94% for LR/RF/XGB, ≈ 46% on TabPFN. Random scoring lower than genetic is *expected* (blind sampling is a weaker optimizer than directed evolutionary search), and your text already concedes it. The problem is the implication: **you gave DiCE its weaker optimizer on its harder task** — a fairness objection independent of the categorical question, and almost certainly what is really behind R3's "makes no sense." So "random because categorical" is doubly weak: unsupported *and* it looks like the baseline was handicapped where the comparison matters most.
The two-part defensible argument (replaces the one-line justification):
1. **Genetic was not a clean option on mixed data** — DiCE-genetic's documented one-hot failures make random the *reliable* choice, not a chosen handicap. Needs your own logged evidence.
2. **The optimizer choice does not drive the headline conclusion** — DiCE's TabPFN collapse appears to show up under *genetic* on the synthetic side too, which would decouple "DiCE fails on TabPFN" from the optimizer choice. **⚠ But the "≈65% under genetic" figure is currently invalid: E5b is a confirmed bug — DiCE's TabPFN runs generated against RF, not TabPFN.** Do not use this datapoint until E5b is fixed and the run regenerated. After the fix, either re-establish this point on corrected numbers, or — if you drop DiCE+TabPFN (E5b option a) — base the argument on the other valid evidence instead.
The nuance behind part 1: at the API level DiCE-genetic is *documented* to support mixed data (it has a `categorical_penalty` param; the docs demo it on the mixed Adult set), so arguing "DiCE-genetic can't do categoricals" in the abstract will lose — the docs contradict it. But DiCE-genetic is *practically* unreliable on mixed data (documented issues: `"Feature ... has a value outside the dataset"` with categorical + genetic; invalid multi-level one-hot flips, e.g. green 0→1 and blue 0→1 for one instance; searches that get stuck). Note the *class* of evolutionary CF methods does handle mixed data — MOC does, via MIES + Gower — but you are not benchmarking MOC (E1 decision); that fact lives in related work, not as a run. So the defensible line is: DiCE-genetic is used where it is sound (continuous synthetic sets) and DiCE-random on the mixed real-world sets, with your own logged failures backing that split. Settle it with measurement, not a capability argument.
*Spec (CODE):*
- Run real-world DiCE with the **genetic** optimizer under your protocol (same budget, same instances). Log outcomes per the E2 status schema: `{success, no_flip, no_output/timeout, invalid_onehot}`. Capture concrete failures (invalid one-hot rate, "value outside dataset" errors, wall-clock) with your DiCE version pinned.
- **Verify the failure modes still reproduce on the exact DiCE version you benchmark** — versions change; don't cite behaviour you haven't re-confirmed.
- **Pull the exact per-classifier completeness values** for DiCE-genetic (synthetic) and DiCE-random (real-world) from your results CSVs, to replace the approximate ≈65% / ≈46% figures read off the bar charts. You'll cite these in the decoupling argument.
- **Robustness check (decouples the claim from the optimizer):** confirm quantitatively that DiCE's TabPFN weakness is present in *both* settings — genetic-on-synthetic and random-on-real-world — so the conclusion doesn't rest on the optimizer choice. If the genetic-on-real-world run (above) succeeds, add it as a third datapoint; if it fails, that failure is itself the evidence.
*Spec (TEXT), Section 4.2:*
- If genetic runs cleanly → use it and delete the old justification.
- If it fails/produces invalid CFs/times out → report *that* with your own numbers as the reason for DiCE-random on mixed sets. Point to the related-work discussion (R1w) noting that the evolutionary method purpose-built for mixed data is MOC, and to E1b for why it isn't benchmarked (scaling + availability) — so the response answers R3 without requiring a MOC run.
- **Add one or two sentences stating the split is not a handicap:** DiCE-genetic is used where it is sound (continuous synthetic), DiCE-random where genetic is unreliable (mixed real-world), and DiCE's TabPFN collapse appears under *both* optimizers — so the comparison is not rigged against DiCE and the headline finding is optimizer-independent. This is the sentence that pre-empts the fairness objection.
- Keep DiCE-genetic on the synthetic (continuous-only) sets, where it is appropriate — state this split explicitly so the method choice looks principled, not arbitrary.
- Note (already in your text) that DiCE completeness can be improved by parameter tuning — keep this honest caveat so you're not accused of under-configuring the baseline.

**E5b [CODE/TEXT] — CONFIRMED BUG: the DiCE-TabPFN stale-explainer issue. Must be corrected.** (integrity; affects R3 efficiency + fairness)
**Status: confirmed by the authors — this notebook (`tabpfn_cf5-2.ipynb`) produced the submitted figures.** So the DiCE-TabPFN results in the paper are affected and this is a correctness fix, not a hypothetical.
**What the code does (the bug).** In `run_benchmark`, the DiCE explainer is only (re)built `if model_name != 'TabPFN'`, and `fit_models_for_dice` builds pipelines for LR/XGB/RF only (no TabPFN). Because `models` is ordered LR→XGB→RF→TabPFN, when the loop reaches **TabPFN, `dice_exp` still holds the RF explainer** from the previous iteration. The `dice` block then calls `dice_exp.generate_counterfactuals(...)`, so **"DiCE on TabPFN" actually generates counterfactuals against RF**, which are then validated/metered against TabPFN (`predict_fn_proba`/`pp`). The commented-out `# continue` shows the intended skip was disabled.
**What it invalidates.** DiCE's TabPFN completeness collapse in **Fig 2 (synthetic) and Fig 5 (real-world)** is substantially an artifact of **RF counterfactuals failing to transfer to TabPFN**, not evidence that evolutionary/sampling search struggles on TabPFN's surface — which is how Section 4 / the discussion currently explains it. The bug contaminates: (i) the DiCE-TabPFN bars in Figs 2 & 5, (ii) the causal claim in the text about *why* DiCE fails on TabPFN, (iii) E5's decoupling datapoint (the "≈65% under genetic" figure), and (iv) any fairness framing about DiCE under TabPFN. Treat all pre-fix DiCE-TabPFN numbers as invalid.
*Spec (CODE) — do this first; it gates the DiCE-TabPFN numbers, the efficiency framing, and E5:*
- Fix the wiring by choosing one of:
  - **(a) Honest skip:** genuinely exclude DiCE+TabPFN, record it as `not_run` with the reason "no sklearn-compatible TabPFN pipeline for DiCE" under the E2 schema (not as `no_flip`/failure), and remove the contaminated DiCE-TabPFN bars; **or**
  - **(b) Real run:** wrap TabPFN in a sklearn-compatible estimator DiCE accepts (`predict`/`predict_proba`/`classes_`/`_estimator_type`) and generate against TabPFN itself, so the DiCE-TabPFN numbers are meaningful. Preferred if feasible, since it keeps DiCE in the TabPFN comparison.
- Add a guard so the failure mode can't recur silently: assert the DiCE model matches `model_name` before generating (e.g. rebuild `dice_exp` every iteration, or `raise` if `model_name == 'TabPFN'` and no TabPFN-compatible DiCE model exists).
- Rerun the affected cells and **re-pull every DiCE-TabPFN completeness value**; regenerate Figs 2 and 5.
*Spec (TEXT):*
- Correct the text that attributes DiCE's TabPFN drop to evolutionary-search inefficiency; state the corrected interpretation based on the fixed run (or state that DiCE+TabPFN is not evaluated, if you take option (a)).
- After the fix, re-establish E5's decoupling point on corrected numbers (or, if DiCE+TabPFN is dropped, base the "DiCE fails on TabPFN" discussion only on the other, valid evidence).
- No rebuttal-letter spin here: this is fixed in the rerun, not argued around. It need not be foregrounded to reviewers, but the corrected numbers/interpretation must be internally consistent across text and figures.
You already state ℓ2 is in z-normalised + one-hot space and "dimensionless / a proxy". Expand it:
- spell out exactly how continuous features are z-scored and how one-hot categoricals enter the ℓ2 sum;
- acknowledge R1's point that a unit categorical change ≠ a unit continuous change after standardisation, so ℓ2 is not a user cost;
- [CODE] optional robustness check: recompute the headline comparisons under an alternative distance (e.g., normalised L1 / Gower distance for mixed data) and report in an appendix to show conclusions are not metric-artefacts.

---

## Workstream 4 — Statistical analysis (rework)

R3 and ED: tests are "too strong for the evidence," many matched comparisons have small counts, and there is no correction or handling of dependence across datasets/seeds/factuals. Treat this as a redo, not a patch.

**S1 [CODE] — Make the dataset the primary unit of replication.** (R3, ED)
Seeds and factuals within a dataset are *not* independent, so per-instance Wilcoxon over pooled pairs overstates power.
*Spec (primary analysis):* for each (classifier, baseline, ℓ0 stratum), compute a **per-dataset** summary (median Δℓ2 across that dataset's matched pairs), then run a Wilcoxon signed-rank / sign test across the **≤10 datasets**. Report this as the headline. Demote the per-instance tests to descriptive support. This respects the hierarchy R3 is asking for.

**S2 [CODE] — Multiple-comparison correction.** (R3, ED)
*Spec:* define the test family explicitly (all reported comparisons across baseline × classifier × ℓ0). Apply **Holm–Bonferroni** (family-wise) or **Benjamini–Hochberg** (FDR) to the p-values in Table A3 and in the Section 4/6 claims. Add adjusted p-values (`p_adj`) as a column; base all "statistically significant" language on `p_adj`.

**S3 [CODE] — Report effect sizes, not just p-values.** (R3)
*Spec:* alongside each Wilcoxon, report the **matched-pairs rank-biserial correlation** (or median Δℓ2 with its bootstrap CI, which you already have). This lets you claim *practical* superiority even where you soften *statistical* claims.

**S4 [CODE/TEXT] — Suppress or gray out under-powered cells.** (R3)
You already outline n<30 dots; go further: set a minimum-n threshold for any *significance* claim, and in the text stop asserting superiority from cells with tiny n (e.g., the TabPFN-MCCE comparison you note has matches at ℓ0<5 in only 1 dataset — state that as inconclusive, not a loss/win).

**S5 [TEXT] — Rewrite the significance sentences in Sections 4, 5, 6.** (R3, ED)
Replace strong claims ("statistically significantly closer... Pareto dominates") with the corrected, dataset-level results: report where significance survives correction, where it does not, and lead with effect sizes + Pareto dominance rates (which are descriptive and robust).

---

## Workstream 5 — Ablation studies

R1 and R3 both require ablations; ED asks for per-component benefit. Keep compute tractable by running on a **representative subset** (e.g., 4–5 datasets spanning continuous-only, mixed, and the hard Breast Cancer case) × the 4 classifiers, fixed seeds.

**A1 [CODE] — Proposal-mechanism ablation (the key one).** (R1, R3, ED)
*Spec:*
- **Leave-one-out:** three configs, each zeroing one of π_a/π_b/π_n and renormalising the other two. Report Δcompleteness, Δℓ0, Δℓ2 vs the full method.
- **Mixture sweep:** a coarse simplex over (π_a, π_b, π_n) (e.g., grid on {0, 0.25, 0.5, 0.75, 1.0} with sum 1). Report the response surface / a small table.
Expected: this is what turns Fig. 9's descriptive breakdown into a *causal* justification for keeping all three. Tie back to M5.

**A2 [CODE] — Hyperparameter ablations.** (R1, R3)
*Spec:* one-at-a-time sweeps around the Table 1 defaults, reporting completeness/ℓ0/ℓ2/runtime:
- candidate budget **M** ∈ {500, 1000, 2000, 5000, 10000};
- max changed units **s** ∈ {2, 4, 8, 16};
- base noise **σ_base** ∈ {0.1, 0.25, 0.5, 1.0};
- interpolation range **[α_min, α_max]** ∈ {[0.5,1.0], [0.25,1.0], [0.0,1.0]};
- importance exponent **γ** (sharpening) ∈ {0.5, 1, 2};
- **surrogate** ∈ {LR, RF, none/uniform importance};
- **lexicographic order** {ℓ0→ℓ2 vs ℓ2→ℓ0}.
Reuse the Section 5 relaxation harness — it already sweeps M, s, σ, α, so much of this exists; extend to γ, surrogate, and ordering, and report on more than the single Breast-Cancer/TabPFN cell.

**A3 [TEXT] — Ablation write-up + one summary figure/table.** (R1, R3, ED)
Summarise A1/A2 in one table (deltas vs default) and 1–2 sentences per parameter on *why* the default is chosen. This also pre-empts R3's "heuristic choices may drive the results" concern.

---

## Workstream 6 — Figures

**F1 [FIG] — Enlarge the ℓ0–ℓ2 panels of Fig. 2 and Fig. 5.** (R3)
Split each into its own full-width figure, or a 2×2 (per classifier). Current right panels are too small to read.

**F2 [FIG] — Fix Fig. 3 row labels.** (R3)
Text says "each baseline is compared" but every y-axis row reads `nice_spars`. Either relabel rows to the actual baselines compared or correct the caption/text to say only NICE(sparse) had matched pairs at fixed ℓ0 in the synthetic setting (which the body text implies). Make figure and text consistent.

**F3 [FIG] — Redesign Fig. 6.** (R3)
*Spec:* **ℓ0 on the x-axis**, **Δℓ2 on the y-axis**, **four subplots (one per classifier)**. Do not encode "ℓ0<5" only by colour when it isn't an axis — R3 specifically objects that you can't conclude "ℓ0<5" from a plot without ℓ0 as an axis. Keep dot-area ∝ n and gray-out n<30.

**F4 [FIG] — Redesign Fig. 7.** (R3)
*Spec:* too many overlapping points. Make **one panel per dataset** (small multiples); if space-limited, keep a representative subset in the main text and move the rest to the appendix. Alternative R3 offers: a sparsity-vs-proximity scatter for all methods across datasets — consider adding that as the intuition figure.

**F5 [TEXT/FIG] — General polish.** (ED)
Consistent axis labels/units, readable font sizes, colour-blind-safe palette, and captions that state the takeaway. Ensure ℓ0/ℓ2 subscripts render (several appear broken in the current PDF text layer).

---

## Workstream 7 — Trade-off section, limitations & claim tempering

**L1 [CODE/TEXT] — Reframe Section 5 as both/and: concede the limitation, defend controllability with a comparison. (Check first, then proceed.)** (R3)
The reviewer and the paper are making *different* ℓ0 comparisons, and both are true: relative to PACE's **own default** cap of 8, the recovered CEs (median ℓ0 = 11 after raising the cap 8→20) are "many more feature changes" (reviewer's axis); relative to the **baselines**, ℓ0 = 11 on a 30-feature dataset may still be the sparsest option (author's axis). These don't conflict — a recovered CE can be less sparse than PACE-default yet sparser than every baseline. Don't dismiss the comment as "just interpretation": ~half is a fair framing point (conceding it is free), ~half is a comparison the paper asserts but never shows.
*Spec (CODE) — the decisive check, do before writing the framing:*
- Pull, for the Breast-Cancer instances PACE originally failed and then recovered (Fig 11 set), the **ℓ0 of each baseline (NICE base/sparse, MCCE, DiCE) on those same instances**, under TabPFN. (You confirmed this is easy to extract from existing results.)
- Compare PACE-relaxed median ℓ0 (= 11) against the baseline ℓ0s on the identical instances.
- **Branch on the result:**
  - **If PACE-relaxed is clearly lower** → you have a demonstrated result, not an assertion. Use it: "even when relaxed to recover completeness, PACE remains sparser than all baselines on these instances." This is the evidence that keeps the controllability framing credible.
  - **If baselines are comparable or lower** → "still much lower" does not hold; drop that claim and lean harder into the limitation framing (below). Better to find this now than to have R3 find it.
*Spec (TEXT), Section 5 — write after the check:*
- **Concede first (answers R3 directly):** state plainly that on high-dimensional, continuous-only data with highly non-linear classifiers (Breast Cancer + TabPFN), PACE cannot simultaneously guarantee completeness *and* its default sparsity — a limitation of the default configuration. This costs nothing and buys credibility.
- **Then defend with the comparison (if it held):** present the recovered-instance ℓ0 result so controllability is backed by data, not adjectives — reframe relaxation as "dial completeness up and still be sparser than everyone else," not "abandon sparsity for completeness."
- **Anchor to the formalism (M1):** reference the formal feasibility/actionability definitions — raising the feature-change cap enlarges the feasible set; actionability degrades monotonically as ℓ0 grows. This is what R3 means by "connected back to the formal definition."
- Keep the honest note that ℓ0 = 11 is a *median* over the hardest (recovered) subset, not PACE's typical CE — the 93.9% default successes are ≤ 8 by construction.

**L2 [CODE/TEXT] — Broaden the trade-off analysis beyond the single recovery case.** (R3)
R3 wants "expanded beyond a narrow recovery case." You currently have two: Breast Cancer + TabPFN, and the d=80 synthetic case in Appendix B. Add at least one or two more from the incomplete cells your own Table 4 already identifies — **HELOC + TabPFN (11.2%), HELOC + RF (10.4%), ILPD + TabPFN (9.0%), or Australian + TabPFN (7.2%)** — so the recovery behaviour is shown to generalise rather than being a Breast-Cancer quirk.
*Spec:* [CODE] rerun the five-relaxation ladder (Fig 10 style) + the ℓ0/ℓ2 trade-off (Fig 11 style) on the added case(s); reuse the existing relaxation harness. Where feasible, also pull the baseline-ℓ0 comparison from L1 on these cases. [TEXT] fold a compact summary into Section 5 (full plots to appendix if space-limited).

**L3 [TEXT] — Temper actionability / plausibility / high-stakes claims.** (R3, ED)
In Section 6, state clearly that range + immutability constraints do **not** by themselves establish realistic recourse, causal validity, or domain plausibility. Narrow "particularly suited to high-stakes classification" to a claim about *auditability and explicit trade-off control*, not validated real-world recourse. Remove or qualify the causal-adjacent implications.

**L4 [TEXT] — Add a critical discussion of the training-data requirement.** (R2)
R2 calls the need for training data a major limitation. Add a paragraph: PACE needs the (z-normalised) training set for pools + importance; discuss when this is unavailable/expensive (privacy, deployment-only settings), how it compares to methods needing model access instead, and what a data-free variant would require.

**L5 [TEXT] — Consolidate limitations.** (R2, ED)
Gather L1/L3/L4 plus existing caveats (binary-only, balanced-data-only, narrow baseline set, high-d continuous incompleteness) into one clearly-signposted limitations paragraph/subsection.

---

## Workstream 8 — Minor / global polish

- **X1 [TEXT]** Line numbers on Algorithm 1 (R2) — covered in M3, listed here for the checklist.
- **X2 [TEXT]** References DOI/link pass (R2) — covered in R4w.
- **X3 [TEXT]** Proofread throughout for the presentation issues ED flags generally (broken math subscripts in the current PDF, undefined-then-used symbols, consistency of method names `nice_base`/`nice_spars` vs prose).

---

## Suggested execution order

1. **E5b** (DiCE-TabPFN correctness) — do this *first*: **confirmed bug**, this notebook produced the figures, so the DiCE-TabPFN results are affected. Fix and rerun before anything downstream, since it gates the DiCE-TabPFN numbers, Figs 2 & 5, the efficiency framing, and E5's decoupling argument.
2. **M1–M4, M6** (formalism + notation, incl. verified guidance/target split) — unblocks everything and is cited by every reviewer.
3. **S1–S3** (stats rework) — decides which claims survive; do before rewriting results prose.
4. **E2, E4** (common protocol + TabPFN instrumentation) — needed for uniform tables and the efficiency demonstration; feeds E1b. E4's mechanism is verified, so this is instrumentation + the runtime-vs-calls plot, not discovery.
5. **A1–A2** (ablations) — the other must-have empirical addition.
6. **E1b, E5** (scaling analysis + DiCE-genetic/random split) — scaling analysis replaces the declined MOC run and depends on E4's call counts; E5 depends on E5b being resolved. **E1 is a decision + text task (not running MOC), do it alongside R1w/R2w.**
7. **F1–F4** (figures) — regenerate once the corrected/new numbers exist.
8. **E3, E6, L2** (equal-budget, metric, extra trade-off case) — strengthen.
9. **M5, M7, M8, R1w–R4w, A3, S5, L1, L3–L5, F5, X1–X3** (write-up, related work, tempering, polish).

## Comment → section map (for your response letter)

| Reviewer point | Action item(s) |
|---|---|
| R1: method detail, importance (local/global, SHAP?), interpolation, scoring, lexicographic order | M1, M3, M4, M6 |
| R1: undefined notation (π_a,π_b,π_n; Algorithm 1) | M4 |
| R1: normalisation & ℓ2 comparability | E6 |
| R1: ablation missing (esp. π) | A1, A2 |
| R1: motivation of three mechanisms | M5, A1 |
| R2: restructure Section 3, add flow diagram, subsections | M1, M3, M8 |
| R2: compare to multi-objective/evolutionary (Dandl) | R1w, E1 (decline+justify), E1b (scaling analysis) |
| R2: training-data limitation + critical discussion | L4, L5 |
| R2: line numbers on Algorithm 1; DOI/links | M3/X1, R4w/X2 |
| R3: formal problem before algorithm; per-stage subsections | M1, M3 |
| R3: define proposal-based modestly; temper "foundations" | M2, R3w |
| R3: taxonomy vs sampling/retrieval/generative/heuristic | R3w |
| R3: MOC, Mothilal, Tsiourvas, Marango references | R1w, R2w |
| R3: guidance/surrogate model unclear; baselines use surrogate? | M6, E4 |
| R3: define interpolation, α | M4 |
| R3: heuristic choices need ablation | A1, A2, A3 |
| R3: multiple CE? | M7 |
| R3: common evaluation protocol / uniform completeness | E2, E3 |
| R3: overstrong statistics, no correction/hierarchy | S1, S2, S3, S4, S5 |
| R3: DiCE random vs genetic justification | E5, E5b (correctness) |
| R3: efficiency not rigorously demonstrated (calls, batching, budget) | E4 (verified mechanism + runtime-vs-calls plot), E3 |
| R3: do baselines use the surrogate or the target? | M6 (only PACE uses LR guidance; baselines use target) |
| R3: Figs 2/5 small; Fig 3 labels; Fig 6 axes; Fig 7 overlap | F1, F2, F3, F4 |
| R3: Section 5 as limitation; expand; tie to feasibility | L1, L2, M1 |
| R3: temper actionability/plausibility/high-stakes | L3, L5 |
| ED: method definition/formalisation/motivation/notation | M1–M6 |
| ED: figures, limitations, missing references, polish | F1–F5, L5, R1w–R2w, X3 |
| ED: multi-objective comparison; protocol; ablations; statistics | E1, E1b, E2, A1–A3, S1–S5 |
