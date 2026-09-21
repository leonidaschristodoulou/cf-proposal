# PACE Revision — Evidence Log

Durable record of what's been investigated, fixed, and found while working through
`PACE_revision_action_list.md`. The Claude Code plan file this work was tracked under
lives outside the repo and isn't a permanent record — this file is. Organized by
action-list item. Every claim below is backed by a script, a SLURM job log, and/or a
result file in this repo, referenced by name/ID so it can be re-checked.

---

## E5b — DiCE-TabPFN stale-explainer bug (CONFIRMED, FIXED)

**Root cause** (`tabpfn_cf5.ipynb`, was in cell 8 `fit_models_for_dice` and cell 15
`run_benchmark`): `fit_models_for_dice` only built DiCE pipelines for `["LR","XGB","RF"]`
(no TabPFN), and `run_benchmark` only rebuilt `dice_model`/`dice_exp`
`if model_name != 'TabPFN'`. So when the outer per-model loop reached TabPFN, `dice_exp`
silently kept RF's explainer from the previous iteration — "DiCE vs TabPFN" was actually
generating counterfactuals against RF, then scoring them against TabPFN. A disabled
original safeguard was still sitting in the code (`#print("...takes too long..") / # continue`),
confirming this was once handled correctly and the skip was later disabled without
fixing the staleness.

**Fix applied to the master notebook** (`tabpfn_cf5.ipynb`, verified via `git diff`):
1. `fit_models_for_dice` now also builds a TabPFN pipeline.
2. `run_benchmark` now always rebuilds `dice_model`/`dice_exp` per `model_name` (removes
   the special case entirely) plus an `assert dice_exp.model.model is dice_spec['models'][model_name]`
   guard against the bug recurring silently.

**Feasibility confirmed via bounded tests before committing to a full rerun**
(all in `_e5_lib.py`, an extracted-and-patched copy of the affected notebook cells):
- `e5b_test_dice_tabpfn.py`/`.sbatch` (job 246704): real-world, DiCE-"random", TabPFN,
  `blood_transfusion`, 5 factuals — 5/5 succeeded, ~1.7–1.9s/instance. Tractable.
- `e5_test2_dice_genetic_tabpfn_synth.py`/`.sbatch` (job 246707): synthetic,
  DiCE-"genetic", TabPFN, 1 dataset (5 features), 3 factuals — 3/3 succeeded,
  ~3.7–4.1s/instance. Tractable.
- `e5_test4_dice_genetic_tabpfn_highdim.py`/`.sbatch` (job 246710): same but 80 features
  (largest in the mock suite) — ~5.0–5.1s/instance. Confirms genetic's cost scales mildly
  with dimensionality, not explosively.

**Full regeneration** (writes to new files, never touched
`output_realdata.joblib`/`output_mockdata.joblib` directly — merging is a separate,
still-pending step):
- **Synthetic** (`e5b_regen_dice_tabpfn_mock.py`/`.sbatch`, job 246717): **COMPLETE.**
  Reproduces `tabpfn_cf5.ipynb` cell 20's exact `generate_make_classification_suite`
  params — verified the 30 dataset names match the existing `output_mockdata.joblib`
  byte-for-byte before running. 30/30 datasets, 3000/3000 rows, 3.85h wall clock.
  Output: `dice_tabpfn_corrected_mock.joblib`. **Merged into `output_mockdata.joblib`
  (confirmed 2026-09-05, checked directly against the live file, not just the
  standalone correction file):** 60000 total rows, 3000 `dice`+`TabPFN` rows,
  **99.2% completeness**, `time_s` median 3.65s (mean 4.05s, max 202s — a much
  smaller tail than the real-world side's `diabetes_binarized` outliers, but still
  clearly genuine TabPFN-under-genetic cost, not the old RF-speed contamination —
  consistent with test 2's earlier ~3.7–5.1s/instance finding). Unlike the real-data
  side, no old/new timing comparison was done here (no accessible pre-merge backup
  for the mock file specifically), so there's no "old numbers understated cost by
  Nx" finding to report for synthetic — only that the corrected numbers themselves
  are now confirmed live in the merged file.
  **Per-dataset breakdown (all 30 synthetic datasets, confirmed 2026-09-05):**
  completeness ranges 93–100% with no dataset below 93% (much narrower spread than
  the real-world side's 46.6–100%); mildly counterintuitive trend where completeness
  is *highest* (100% across all 7 datasets) at the largest dimensionality (d=80) and
  most variable at d=5 (93–100%). Timing scales cleanly and modestly with feature
  count: median ~3.5s for d=5 through d=40, stepping up to ~5.0s at d=80 — no
  catastrophic jumps like the real-world side. **Only one genuine outlier in the
  entire 3000-row set**: `mc_f5_red2_inf2_seed1466836456` (max 202s, a single
  instance) — accounts for the overall `time_max=202s` noted above. Contrast with
  `diabetes_binarized`'s 39 outliers on the real-world side — synthetic data's
  DiCE+TabPFN behavior is far better-behaved, likely because it lacks whatever
  real-world data characteristics (missingness, class imbalance, etc.) drive the
  real-world tail-cost pattern.
- **Real-world** (`e5b_regen_dice_tabpfn_real.py`/`.sbatch`): reproduces the 10
  real-dataset names from cell 19 × seeds `[1,3,5,7,9]` × 100 factuals (4870 rows,
  matches `output_realdata.joblib` exactly). Checkpointed per (dataset, seed) pair.
  - First attempt (job 246716) hit its 6h cap at 1100/4870 rows. Cause: **rare
    outlier calls** — 4 instances within `diabetes_binarized`/seed=1 each took
    ~2000s (~33 min) instead of the typical ~1.7–2.3s, a ~1000x spike, though all 4
    still returned `status='ok'` (DiCE eventually found a valid CF). Worth keeping
    for the write-up: a real, reproducible example of a baseline method's
    catastrophic-tail cost against TabPFN, useful color for E1b/E4 beyond the
    call-count table.
  - Resubmitted (job 249821) with the partition's max walltime (`23:59:00`),
    resuming from checkpoint. **Also hit the 24h cap** — but got much further
    (41/50 pairs, 3970/4870 rows) before running out. Root cause, now confirmed:
    `diabetes_binarized`'s ~2000s outliers aren't a seed=1 fluke — **they occur in
    all 5 seeds**, ~39 instances total (~8% of its 500), each ~2000-2080s
    (~1200x the normal ~1.7s). That alone ate ~22 of the ~24 available hours.
    Strong, reproducible, dataset-specific finding worth citing precisely in the
    write-up (E1b/E4) — not a fluke, a consistent ~8% catastrophic-tail rate.
    `diabetes_binarized` is now fully done; only 9 pairs remain
    (`blood_transfusion` seeds 3/5/7/9, `chronic_kidney_disease` all 5 seeds),
    which should go much faster now that the worst offender is behind it.
    Resubmitted again as **job 267328** — **COMPLETE**, only 0.92h for the
    remaining 9 pairs (`chronic_kidney_disease` did not repeat
    `diabetes_binarized`'s pathology — good news, confirms it's dataset-specific
    not systemic). Output: `dice_tabpfn_corrected_real.joblib`, **full 4870/4870
    rows, all 10 real datasets × 5 seeds — the entire real-world DiCE+TabPFN
    correction is now done.**

  **Corrected completeness by dataset (DiCE-random, TabPFN, post-E5b-fix):**
  | dataset | completeness | outliers (>60s) |
  |---|---|---|
  | australian | 100.0% | 0 |
  | blood_transfusion | 100.0% | 2 |
  | heart_disease | 100.0% | 0 |
  | credit-g | 99.8% | 0 |
  | diabetes_binarized | 99.2% | 39 |
  | heloc | 97.6% | 0 |
  | breast_cancer | 87.4% | 0 |
  | chronic_kidney_disease | 82.0% | 2 |
  | sick | 71.2% | 0 (excluded from genetic argument, see above — same NaN issue) |
  | ilpd | 46.6% | 0 |

  **Timing also changed, not just completeness — checked directly against your
  backup at `/nvme/h/lchristodoulou/data_p318/pace_results/output_realdata.joblib`.**
  All other methods'/models' `time_s` are byte-identical between old and new
  files (confirms the merge touched nothing else). But `dice`+TabPFN's timing
  genuinely shifted: median 0.85s → 1.87s (**~2.2x slower**, ranging 2.0x–3.5x
  by dataset), mean 2.7s → 19.9s (~7.4x, tail-driven), max 366s → 2079s. This
  makes sense in hindsight: the old numbers were measuring `dice`+RF's search
  cost (median 0.41s) plus one genuine TabPFN call for final validation
  (~0.4s) ≈ 0.85s — the only part of the old pipeline that actually touched
  TabPFN. **The original runtime comparison (Timing box-plot, any "PACE is Nx
  faster under TabPFN" claim) understated DiCE's true cost under TabPFN** —
  the corrected numbers make PACE's efficiency advantage larger, not smaller,
  but the timing figure and any quoted runtime ratios need regenerating from
  the corrected data before being cited.

  **Major narrative implication — worth flagging prominently for the E5b text
  and Section 5 discussion.** The pre-fix (buggy, stale-RF-explainer) numbers
  made DiCE+TabPFN look uniformly weak (~46–65% range cited in the action
  list). The corrected numbers tell a different story: **DiCE+TabPFN actually
  performs very well on most datasets** (100% on 3 of them, >97% on 6 of 10) —
  the bug was validating RF-optimized counterfactuals against TabPFN's
  different decision surface, which artificially depressed completeness
  across the board. `ilpd` (46.6%) stands out as a genuinely hard case, with
  `chronic_kidney_disease` (82.0%) and `sick` (71.2%, though `sick`'s issue is
  the unrelated NaN-validation problem, not TabPFN-specific) as secondary
  soft spots — but the "DiCE collapses on TabPFN" framing the manuscript
  currently uses does **not** hold uniformly anymore. The E5b text task
  ("correct the text that attributes DiCE's TabPFN drop to evolutionary-
  search inefficiency") needs a more nuanced rewrite than originally
  anticipated — not just correcting the *cause*, but correcting the
  *magnitude and scope* of the drop itself. `diabetes_binarized`'s 39
  outliers keep its raw completeness high (99.2%) despite the catastrophic
  tail cost — worth being precise in the text that "high completeness" and
  "cheap/fast" are not the same claim there.

**Remaining step**: once the real-world regen finishes, inspect both corrected files,
then merge into `output_realdata.joblib`/`output_mockdata.joblib` (drop old
contaminated DiCE+TabPFN rows for these pairs, append corrected ones) — this is the
one step that needs explicit sign-off before running, since it overwrites the shared
result caches everything downstream reads from.

---

## E5 — DiCE genetic-vs-random justification (evidence gathered)

**Original problem**: the manuscript uses DiCE-genetic only on synthetic data and
DiCE-random only on real-world data, with no head-to-head comparison and no logged
evidence for why — just an assertion. Reviewer flagged this as looking like a handicap
(giving DiCE its weaker optimizer on its harder task).

### Part 1 — was genetic a clean option on mixed/real-world data?

`e5_test3_dice_genetic_realworld_failures.py`/`.sbatch` (job 246708, pilot) then
`e5_regen_dice_genetic_lr_realworld.py`/`.sbatch` (job 250457, full sweep — **COMPLETE**,
0.90h, 4870/4870 rows, output `dice_genetic_lr_realworld.joblib`): DiCE-genetic forced
onto all 10 real-world datasets against LR (LR chosen because genetic's categorical/
search-validity issues are optimizer-level, not classifier-level — no need to repeat
against RF/XGB/TabPFN for this specific question).

**Completeness by dataset** (genetic, LR):
| dataset | completeness |
|---|---|
| australian | 98.6% |
| breast_cancer | 96.6% |
| chronic_kidney_disease | 95.6% |
| diabetes_binarized | 96.4% |
| heart_disease | 95.1% |
| credit-g | 86.2% |
| blood_transfusion | 79.4% |
| ilpd | 77.0% |
| heloc | 71.6% |
| sick | 0.0% (excluded — see below) |

**Three distinct, reproducible genetic-specific failure mechanisms identified**
(not just one — meaningfully stronger than a single-dataset pilot):
1. **`credit-g`**: generic exhausted-search failures (`"No counterfactuals found for
   any of the query points! Kindly check your configuration."`).
2. **`blood_transfusion`** (no categorical features at all — proves the fragility
   isn't purely a one-hot issue): `"empty range for randrange()"` — an internal crash
   when a feature's computed permitted range degenerates to zero width.
3. **`sick`** — see below; ultimately excluded, not a genetic-specific mechanism.

**`sick` (0% under genetic) — investigated and excluded from this argument.**
Traced to `dice_ml`'s `explainer_base._validate_counterfactual_configuration`
(`~/.conda/envs/tabpfn/lib/python3.13/site-packages/dice_ml/explainer_interfaces/explainer_base.py:76-80`),
a NaN check on the query instance shared by both optimizers (neither `dice_genetic.py`
nor `dice_random.py` override the base class's `generate_counterfactuals`). Confirmed
via direct A/B test that both raise the identical `UserConfigValidationException` on
the same NaN-containing query instance. So this is **not genetic-specific** — ruled
out as evidence for this argument. Whether the original random-based 71-72%
completeness on `sick` still reproduces under the current codebase is a separate,
legitimate reproducibility question that was **not pursued further** (correctly
flagged as scope creep into debugging `dice_ml`'s internals, which isn't ours to fix).
**Decision: cite `credit-g` and `blood_transfusion` as the two genetic-failure
examples for E5; do not cite `sick`.**

### Part 2 — robustness/decoupling check (E5's "third datapoint")

The point: confirm DiCE's TabPFN weakness shows up under *both* optimizers, so the
"DiCE fails on TabPFN" conclusion doesn't rest on which search method was used.
Genetic-on-synthetic-TabPFN already exists (post-E5b-fix, see above);
random-on-real-world-TabPFN is being regenerated (see above); this adds the missing
genetic-on-real-world-TabPFN datapoint.

`e5_regen_dice_genetic_tabpfn_realworld.py`/`.sbatch` (job 250458): **COMPLETE**,
6.52h, 1500/1500 rows, scoped to `credit-g`/`australian`/`sick` (the 3 real-world
datasets with the clearest categorical structure — RF/XGB not run, not required by
E5's argument). Output: `dice_genetic_tabpfn_realworld.joblib`.

**Completeness by dataset** (genetic, TabPFN):
| dataset | completeness |
|---|---|
| australian | 96.8% |
| credit-g | 89.0% (slightly *higher* than under LR-genetic, 86.2%) |
| sick | 0.0% (expected — same shared NaN-validation issue) |

**Comparison now resolved — real-world regen (job 267328) finished.** The
robustness-check question was "does DiCE's TabPFN weakness show up under both
optimizers." The corrected data changes the premise:

| dataset | random+TabPFN | genetic+TabPFN |
|---|---|---|
| australian | 100.0% | 96.8% |
| credit-g | 99.8% | 89.0% |
| sick | 71.2% | 0.0% (excluded, shared NaN issue) |

On the two clean datasets, **both optimizers do well against TabPFN** (high
80s–100%) — random is a few points *better* than genetic, not dramatically
worse. This is the opposite of what the original decoupling check was designed
to find (it expected both to show a comparably large TabPFN-specific
collapse). Since there isn't a large TabPFN collapse to decouple from the
optimizer choice on these two datasets post-fix, **this specific "third
datapoint" argument needs reframing, not just reporting** — the corrected
data doesn't support "DiCE fails on TabPFN regardless of optimizer" as
strongly as hoped; it more supports "DiCE-genetic is somewhat less reliable
than DiCE-random in general (consistent with the broader genetic-fragility
findings above), and neither optimizer collapses dramatically against TabPFN
on datasets without genetic's other failure modes." Worth discussing before
writing the E5 section — this is a genuine finding, not a setback, but it
changes what E5's text should claim.

---

## E2 — NICE-sparse `not_run` characterization, and uniform completeness/reliability (real data: DONE)

**Original problem**: `run_benchmark`'s `_nice_spars_skip` hardcodes 5
`(TabPFN, dataset)` skip entries — `diabetes_binarized`, `australian`,
`blood_transfusion`, `chronic_kidney_disease`, `heloc` — with **no logged reason
anywhere**. Confirmed via code inspection there's no equivalent staleness bug to
E5b's here (NICE/MCCE explainer objects are unconditionally rebuilt fresh every
outer-loop model iteration, unlike DiCE's old bug) — so each skip is either a
deliberate-but-undocumented exclusion or an unverified assumption.

**v1** (`e2_test_nice_spars_tabpfn_skipped.py`/`.sbatch`, job 250459): bypassed the
skip list directly, 5 factuals/dataset, try/except per dataset. Ran out its full 3h
cap without finishing — a hang can't be caught by try/except. Findings before it
died:
- `diabetes_binarized` + TabPFN: **works fine**, 5/5 succeeded, ~1.2–2.5s/instance.
- `australian` + TabPFN: **genuine hang** — explainer built in 0.2s, then the first
  `nice_cf_one` call never returned for ~2h40min until the wall-clock killed it.
- Never reached the remaining 3.

**v2** (`e2_test_nice_spars_tabpfn_skipped_v2.py`/`.sbatch`, job 253261): tested the
remaining 3 with a hard per-dataset timeout (300s) enforced via a `spawn`-context
subprocess (`process.terminate()`/`.kill()` from the parent — reliably kills a hang,
unlike try/except). **COMPLETE**, ~15 min total (3 × 300s cap). Output:
`nice_spars_tabpfn_skipped_result_v2.csv`.

**Correction: `_nice_spars_skip` has 11 entries, not 5 — the other 6 are RF/XGB, not
TabPFN, and were not investigated above.** Built a method-variant summary table
(now in `cfprop_plots.ipynb`, new "## 0" section, empirically checking row presence
per dataset/classifier rather than trusting the hardcoded list) — confirms the
skip list is accurate (no undocumented gaps) but incomplete in scope: `RF` is also
skipped on `australian`/`diabetes_binarized`/`heloc`, and `XGB` on
`australian`/`blood_transfusion`/`heloc`. `e2_regen_nice_spars_rf_xgb_unskipped.py`/
`.sbatch` (job 304976, `cpu` partition — no GPU needed for RF/XGB) applies the same
persistent-worker + 120s-per-instance-timeout design to these 6 combos, full standard
protocol (5 seeds × 100 factuals). **COMPLETE** (job 304976, `cpu` partition, all
30/30 (model,dataset,seed) pairs, 3000/3000 rows). Output:
`nice_spars_rf_xgb_unskipped.joblib`.

**Result: unlike the TabPFN skips, all 6 of these turn out safe, not hung.**
| model | dataset | completeness | notes |
|---|---|---|---|
| RF | australian | 99.2% | |
| RF | diabetes_binarized | 98.6% | |
| RF | heloc | 96.4% | most stall-prone (1,1,1,6,1 across 5 seeds), still completes |
| XGB | australian | 99.8% | |
| XGB | blood_transfusion | 99.4% | |
| XGB | heloc | 99.8% | |

Only 16/3000 (0.5%) timeouts overall, 18 genuine `not_flipped`. XGB pairs were almost
uniformly fast (~8-10s per 100 instances); RF pairs slower and more stall-prone but
still completed cleanly. **These 6 skip entries look like they were excessively
conservative** — probably added from isolated slow runs during interactive
development, same undocumented-ad-hoc pattern as the rest of `_nice_spars_skip` —
not genuine non-termination. Combined with `diabetes_binarized`/`heloc` on TabPFN
(82.2%/90.0%, resolved earlier), **8 of the original 11 skip entries now have real
completeness numbers and should be un-skipped**; only the 3 genuine TabPFN hangs
(`australian`, `blood_transfusion`, `chronic_kidney_disease`) remain legitimate
`not_run`. NICE-sparse's E2 characterization is now complete across the full
`_nice_spars_skip` list.

**Full picture across the original 5 TabPFN-specific combos:**
| dataset | result | verdict |
|---|---|---|
| `diabetes_binarized` | works fine, ~1.2–2.5s/instance | skip looks unnecessary |
| `australian` | genuine hang, never returns | skip justified |
| `blood_transfusion` | genuine hang — even explainer *construction* never completes | skip justified |
| `chronic_kidney_disease` | same — hangs during explainer construction | skip justified |
| `heloc` | works, but slow (10–20s/instance vs. ~0.06–2s elsewhere) | cap-sensitive — see below |

**Methodological note (important — read before acting on the table above):** the
temptation is to now decide "run NICE-sparse for real on `diabetes_binarized` (and
maybe `heloc`), keep skipping the other 3" as five separate per-dataset judgment
calls. **That would recreate exactly the kind of undocumented, inconsistent decision
that produced the original unexplained skip list.** The principled fix is E3's
"identical wall-clock timeout applied to every method" — one fixed cap, applied
mechanically to every method/dataset/model combination in `run_benchmark` (not just
NICE), replacing `_nice_spars_skip`/`_nice_base_skip` entirely.

**Decided: 120 seconds per instance** (see below). At full scale under that cap
(see the completed regen further down), `diabetes_binarized` and `heloc` both turn
out to be **partial-completeness cases with a real, non-trivial timeout rate**
(82.2% and 90.0% respectively), not simply "pass" as the initial 5-instance sample
suggested — a good illustration of why the small diagnostic sample wasn't enough
on its own and the full-protocol run mattered. `australian`/`blood_transfusion`/
`chronic_kidney_disease` still fail entirely under any reasonable cap (don't
terminate even after 300s).

### Continued: uniform completeness/reliability across all methods (E2's full scope)

Everything above characterizes NICE-sparse's `not_run` entries specifically —
an input to E2, not E2 itself. E2's actual ask (R3/ED's exact wording,
obtained this session): *"The empirical evaluation is useful but not
sufficient to validate the proposed methodology. The comparison with
baselines is difficult to interpret because failures, timeouts, omitted
configurations, no-output cases, and no-flip cases are not handled under a
single common denominator with identical budgets. Completeness and
reliability should be reported uniformly for all methods."* No standalone
reviewer letter/decision file exists anywhere in this repo or found
system-wide — this quote (and the other short quoted fragments elsewhere in
`PACE_revision_action_list.md`, e.g. `"makes no sense"`) is the only verbatim
reviewer text available; everything else in the action list is a paraphrase.

**Two distinct problems, both real, found by inspecting the actual data
before writing any fix.**

1. **Non-common denominator, structural.** `output_realdata.joblib` row
   counts by method: `pace`/`nice_base`/`dice`/`mcce` all had exactly 19480
   rows (the full protocol); `nice_spars` had 17980 — missing exactly 1500,
   which is precisely the 3 genuinely-hung TabPFN combos
   (`australian`/`blood_transfusion`/`chronic_kidney_disease`) × 5 seeds ×
   100 factuals. These weren't rows with a `timeout` status (151 of those
   already existed correctly, from other combos that were attempted and hit
   the cap) — they were **entirely absent**, because `_nice_spars_skip`
   excluded them from the main benchmark run before it started, based on
   the diagnostic evidence above. The existing flip-rate figure computes
   `groupby(["model","method"])["flip_ok"].mean()` over whatever rows
   exist per method — so `nice_spars`'s TabPFN rate was silently averaged
   over 7 datasets instead of 10, inflating it by excluding exactly the
   hardest cases.

2. **Non-uniform status labels.** Every method already had its own `status`
   column, just with method-specific vocabulary rather than R3's shared one:
   `pace`/`dice`: `ok`, `no_flip`; `nice_base`/`nice_spars`: `ok`,
   `violates_immutable`, `not_flipped`, (`timeout` for sparse only);
   `mcce`: `ok`, `no_cf`, `flip_failed`. Checked whether `l0`/`l2` being
   populated could help disambiguate which E2 bucket each maps to —
   **it can't**: every failure category, across every method including
   PACE's own `no_flip`, has null `l0`/`l2`. So the mapping is a genuine
   semantic judgment call, not a mechanical relabel.

**Fix 1 — added the missing rows.** `e2_add_notrun_rows.py` (not committed,
one-off data-repair script): copied the 1500 exact `(dataset, model, seed,
idx)` keys from `pace`'s rows for the same 3 combos (guarantees correct key
alignment rather than reconstructing seeds/indices by hand), appended them
to `nice_spars` with `status="not_run"`, all outcome columns `NaN`, and
static per-(dataset,model) metadata (`clf_acc`, `clf_auc`, `n_cols_model`,
`n_rows`) backfilled from sibling `pace` rows — same convention as the
earlier NICE-sparse un-skip merge. Verified before writing: confirmed zero
existing `nice_spars` rows at these exact keys (pure addition, no
collision risk). Backed up first: `output_realdata.joblib.pre_e2_notrun`
(both in the repo and in the session scratch dir). Post-merge: every method
now has exactly 19480 rows. **This alone fixed the existing flip-rate
figure with no code change** — `real_flip_rate.pdf` regenerated and
verified: `nice_spars`/TabPFN visibly drops from its previous
(silently-inflated) bar to $\approx$0.53, matching the corrected 54.1%
completeness computed below.

**Fix 2 — unified status vocabulary + tables.** Added to
`cfprop_plots.ipynb` (new section "2b", right after the existing flip-rate
figure): an `E2_STATUS_MAP` dict (`ok`$\to$`success`; `no_flip`/
`not_flipped`/`flip_failed`$\to$`no_flip`; `dice_invalid_config`/
`no_cf`$\to$`no_output`; `violates_immutable`$\to$`invalid`; `timeout`
unchanged; `not_run` unchanged), applied with an assertion that no status
value is left unmapped. `e2_completeness_table` (per method/dataset/model,
common denominator = full protocol size for every cell, breaking out every
E2 status bucket's count) and `e2_reliability_table` (method × classifier,
pooled over datasets — the direct baseline-analogue of the paper's existing
PACE-only reliability table) both exported as tidy CSVs
(`real_e2_completeness.csv`, `real_e2_reliability.csv`).

**Verified end-to-end** (isolated `jupyter nbconvert --execute` pass,
`data_in="real"`, cells 0 through the new section, not saved into the
tracked notebook — same verification pattern as Workstream 4). Key numbers
from the reliability table: `nice_spars`/TabPFN completeness under the
common denominator is **54.1%** (2633 success / 4870 attempted per
classifier, pooled over 10 datasets) — sharply lower than any single-dataset
number reported elsewhere in this log, because it now correctly folds in
the 3 always-failing combos rather than excluding them. `nice_base` is the
most affected method overall by the `invalid` (constraint-violation)
bucket: 37.0% completeness on TabPFN specifically (1284 of those rows are
also `no_flip` on top of 1783 `invalid` — NICE(base)'s lack of an
immutability-enforcement mechanism, already documented in
`draft_nice_base_vs_sparse_note.md`, now has a uniform-schema number behind
it). Full reliability table (all method × classifier cells) is in
`real_e2_reliability.csv`.

**Text status: pasted into `pace.tex`** — resolution note added to the
existing `\revitem{E2}` block (Section~\ref{sec:exp} intro) with the exact
reviewer quote, the two fixes, and the headline 54.1% number.
**Not yet done:** the reliability table itself as a LaTeX table in the
manuscript (baseline-analogue of `tab:pace_completeness`) — the CSV exists,
the table does not; and the mock/synthetic-data equivalent of this fix
(only real data was addressed this pass — worth checking whether mock has
an analogous structural gap before assuming it doesn't).

### Follow-up (same session): the 3 stub `not_run` combos were actually re-run

User asked to properly run the 3 genuinely-skipped combos
(`australian`/`blood_transfusion`/`chronic_kidney_disease` × TabPFN ×
`nice_spars`) under the actual decided 120s-per-instance protocol, rather
than leave them as `not_run` stubs based on the older, looser diagnostics
(an uncapped ~2h40min kill for `australian`; a 300s per-*dataset* kill,
before any per-instance loop, for the other two). Reused the exact
persistent-worker mechanism from the earlier successful
`diabetes_binarized`/`heloc` regen
(`e2_regen_nice_spars_tabpfn_unskipped_v2.py`), adapted as
`e2_regen_nice_spars_tabpfn_remaining3.py`/`.sbatch`: explainer built once
per (dataset, seed), a construction hang bounded to exactly 120s and marks
the *whole* pair `build_timeout` (cheap if it happens), a per-instance
stall bounded to exactly 120s and marks only *that* instance `timeout`
(loses one instance, not the whole pair), checkpointed after every
completed pair so nothing is lost to a wall-clock kill.

**Verified against the code before submitting**, given the cost of a wrong
24h GPU job: confirmed `load_many` resolves through `_e5_lib.py`'s
wildcard-import chain from `get_real_datasets.py` (grep initially came up
empty and looked alarming — false alarm, traced it down before submitting,
not after a failed run).

**Run 1** (job 317604, a100, submitted with the full 24h budget): completed
9/15 pairs (all 5 `australian` seeds, `blood_transfusion` seeds 1/3/5/7)
before running low on its wall-clock budget. **Handoff, mid-session**: user
asked whether to kill and resubmit once the in-flight pair finished, to get
a fresh 24h budget rather than risk the wall-clock cutting off an
in-progress pair (checkpointing only happens *after* a pair completes, so a
mid-pair kill loses that pair's ~1-3h entirely). Set up an unattended
handoff script: poll until pair 9's checkpoint line appears in the log,
`scancel` the old job at that exact boundary, immediately resubmit
(job 318440 picks up the checkpoint and resumes at pair 10), keep polling
the new job. Executed cleanly — zero work lost, ~22.5h saved off what a
wall-clock kill + manual restart would have cost.

**Result: none of the three are genuine non-terminating hangs.** All three
build their explainer successfully (0.5-1.1s) and have real, substantial
completeness:

| dataset | completeness (NICE-sparse, TabPFN, full 500) | status counts |
|---|---|---|
| `australian` | 75/500 = **15.0%** | (hardest of the 3, closest to the old "hang" story, but still real) |
| `blood_transfusion` | 275/500 = **55.0%** | comparable to `diabetes_binarized`/`heloc` |
| `chronic_kidney_disease` | 261/500 = **52.2%** | comparable to `diabetes_binarized`/`heloc` |
| **pooled** | 611/1500 = 40.7% | 611 `ok`, 889 `timeout` |

This is a real, belated correction to the manuscript's own narrative, not
just a data-completeness footnote: the earlier "3 genuinely never
terminate" characterization was itself an artifact of testing under a
budget looser and less methodologically consistent than the one the paper
actually commits to (the whole point of E3's common-cap decision). Under
the cap the paper actually uses, there is no true hang among these three —
only `australian` is dramatically harder than the rest of the suite.

**Merge**: dropped the 1500 `not_run` stub rows for these exact keys from
`output_realdata.joblib` (backed up first as
`output_realdata.joblib.pre_e2_remaining3_merge`), replaced with the real
`ok`/`timeout` rows (`e2_merge_remaining3.py`, not committed), backfilling
static metadata (`clf_acc`/`clf_auc`/`n_cols_model`/`n_rows`) from sibling
`pace` rows as before. Verified: every method still has exactly 19480 rows
post-merge.

**Three more bugs found and fixed while regenerating figures against the
corrected data**, all in `cfprop_plots.ipynb`:

1. **Timing figure was unfiltered by status.** User asked directly whether
   it was filtered to `status=="ok"` — it wasn't. Two distinct problems
   from this: `timeout` rows are pinned at exactly 120.0s (not a
   measurement, an artifact of the cap) and were being plotted as if they
   were real runtimes; and `matplotlib.boxplot` silently invalidates an
   *entire* box (median line becomes `[nan, nan]`, not just dropping the
   NaN entries) when even one `NaN` is present — confirmed directly, not
   assumed — which the `not_run` stub rows' `time_s=NaN` had been doing to
   `nice_spars`/TabPFN's box specifically (it rendered completely blank in
   the previously-committed `real_flip_rate`... `realdata_times.pdf`).
   Fixed by filtering to `status=="ok"` before extracting `time_s`. Mock
   data has zero `timeout`/`not_run` rows so this is a no-op there,
   confirmed directly rather than assumed.
2. **`nice_base` was missing from the completeness figure.** The global
   `methods` list deliberately excludes `nice_base` on real data for the
   $\ell_0$/$\ell_2$ *quality* comparisons (correct — see
   `draft_nice_base_vs_sparse_note.md`), but the completeness bar chart
   reused the same list, silently hiding `nice_base`'s completeness —
   exactly the uniform-reporting number E2 exists to surface. Fixed with a
   separate `completeness_methods = sorted(df["method"].unique())` scoped
   to that one panel only; the $\ell_0$/$\ell_2$ scatter panel and every
   quality comparison elsewhere still correctly use the narrower `methods`
   list. With `nice_base` visible: 62.5% (LR), 63.1% (RF), 63.1% (XGB),
   37.0% (TabPFN) — matches the reliability table exactly, confirming the
   fix. Added as an addendum to `draft_nice_base_vs_sparse_note.md` rather
   than a new file.
3. **New figure**: `plot_e2_reliability_stacked` (section "2c" of the
   notebook) — one stacked bar per (model, method), segments = every E2
   status bucket as a fraction of the total, rather than only the binary
   success rate. User's suggestion, not R3's, but squarely in E2's spirit
   (the completeness number alone can't distinguish *why* a method falls
   short; this figure can, at a glance — e.g. `nice_base`'s shortfall
   reads immediately as mostly orange/`invalid`, `dice`'s as mostly
   gray/`no_output`, `nice_spars`/TabPFN's now shows real orange/`invalid`
   and vermillion/`timeout` where it used to show black/`not_run`).
   Output: `real_e2_reliability_stacked.pdf`.

**Post-merge numbers**: `nice_spars`/TabPFN completeness rose from 54.1% to
**66.6%** (3244/4870). Checked whether this materially changes anything
already written up: the hierarchical test result barely moved (38/67
significant, mean ICC 0.140 — was 39/67, ICC 0.146) and the 2×2 figure is
visually unchanged, so Workstream 4's write-up stands without correction.
L1's Breast-Cancer-specific baseline-$\ell_0$ comparison is entirely
unaffected (different dataset). Pareto dominance and the completeness
figures were regenerated and re-sent.

**Text status: pasted into `pace.tex`** — corrected the "on closer
investigation only three genuinely never terminate" sentence in
Section~\ref{sec:exp} directly (with a `\revnote{E2}` marking it corrected
and why), and extended the existing E2 resolution note with the follow-up.
**Not yet done:** same as before — the reliability table and the new
stacked figure aren't placed in the manuscript body yet; mock/synthetic
data's equivalent structural gap (if any) still hasn't been checked.

### Follow-up (same session): Timing figure had DiCE outliers predating the cap

User noticed the Timing figure (`realdata_times.pdf`) had DiCE points above
1000s and asked to find and rerun those instances under the cap. **Checked
first, did not rerun**: all 39 of the >1000s rows are `diabetes_binarized`
+ TabPFN, `status=="ok"`, ~1990–2079s — this is exactly the
"catastrophic-tail" datapoint already cited by name in `pace.tex`
(Section~6/E1b, "corroborated by a real, reproducible catastrophic-tail...")
and in this log's E1b entry (~8% of 500, all 5 seeds, ~1000x the typical
1.7–2.3s). Rerunning and overwriting them under a live 120s cap would have
replaced that exact cited evidence with generic `timeout` rows, destroying
the corroboration E1b's scaling argument relies on — the opposite of what
was asked. Broadened the check to every method: **62 rows total, all
DiCE**, none from any other method (`diabetes_binarized`/TabPFN: 39;
`diabetes_binarized`/RF: 20; `blood_transfusion`/TabPFN: 1;
`chronic_kidney_disease`/TabPFN: 2), all predating the wall-clock cap ever
being implemented as a live per-call mechanism in `run_benchmark`
("Open items" #1 above — decided, but applied post-hoc, not live).

**Fix: applied the already-decided post-hoc rule properly, without
touching the raw data.** Two changes in `cfprop_plots.ipynb`: (1) the
Timing figure's filter gained `& (df["time_s"] <= 120)` alongside the
existing `status=="ok"`, so these 62 rows no longer appear as if they were
representative successful timings under the protocol the figure claims to
represent (verified: `realdata_times.pdf`'s y-axis now tops out ~100s,
was ~2000s). (2) E2's `status_e2` mapping (section "2b") now relabels any
`success` row with `time_s>120` to `timeout` before building the
completeness/reliability tables — printed confirmation: "62 success rows
relabeled timeout for analysis (raw status/time_s unchanged)". **Verified
directly, not assumed**: hashed `output_realdata.joblib` before and after
running the fixed notebook cells — byte-identical, confirming the raw
`status`/`time_s` values (and therefore E1b's citation) are untouched; only
the derived `status_e2` column used for E2's own analysis is affected.
Completeness impact is exactly as the user expected going in ("small, won't
affect the results") — 62 rows out of 97400.

**Text status:** not yet pasted into `pace.tex` — this is a plotting/
analysis-consistency fix with no numbers currently cited in the manuscript
text that need correcting (the only place these rows are cited, E1b's
catastrophic-tail paragraph, reads from the raw data directly and is
unaffected).

---

## Open items / next steps

1. **Common wall-clock timeout: DECIDED — 120 seconds per instance.** Applies
   uniformly to every method (NICE, DiCE, MCCE, PACE) in `run_benchmark`, replacing
   `_nice_spars_skip`/`_nice_base_skip` and any other hardcoded skip logic entirely.
   Consequences under this value, given evidence gathered so far: `heloc`'s NICE-
   sparse (~10–20s/instance) passes with margin — no longer excluded; the 3 genuine
   NICE-sparse hangs (`australian`, `blood_transfusion`, `chronic_kidney_disease`)
   fail regardless (they don't terminate even after 300s); the ~2000s DiCE-random
   outliers on `diabetes_binarized` get reclassified as `timeout` rather than slow
   successes. Not yet implemented as a live per-call mechanism in `run_benchmark`
   (still needed for E2/E3's CODE work) — for already-completed/in-progress runs it
   can be applied post-hoc, since `time_s` is already recorded per row for every
   method (any row with `time_s > 120` gets relabeled `timeout` when building the
   final tables, no rerun needed).

2. **NICE-sparse full-protocol regen for the two un-skipped combos: COMPLETE**
   (v1 revealed a design flaw, v2 fixed it — see final numbers above:
   `diabetes_binarized` 82.2%, `heloc` 90.0%, both legitimate real results).
   - **v1** (`e2_regen_nice_spars_tabpfn_unskipped.py`/`.sbatch`, job 253438):
     used a single 2400s budget for the *whole* (dataset, seed) pair (100
     factuals). Important correction to the earlier small-sample conclusion:
     **every one of the 10 pairs timed out**, completing only 2-30/100 instances
     before stalling at an unpredictable point (not a uniform slowdown — an
     early-or-late catastrophic stall, same character as `diabetes_binarized`'s
     DiCE-random outliers). The 5-factual diagnostic sample that suggested these
     two datasets were "safe" was **not representative** — both datasets have
     their own rare catastrophic-instance problem, just not visible at n=5.
     Root cause of the failed run: a per-*pair* backstop lets one bad instance
     early in the sequence starve out the rest of the pair — doesn't actually
     implement the agreed per-*instance* rule.
   - **v2** (`e2_regen_nice_spars_tabpfn_unskipped_v2.py`/`.sbatch`, job 267329):
     properly enforces a hard 120s deadline **per instance**. A persistent worker
     subprocess builds the explainer once and processes factuals one at a time
     over a Queue; if the parent gets no response within 120s, it kills only that
     worker (losing the one stuck instance, marked `timeout`), spawns a fresh one
     (explainer rebuild is cheap, <1s observed), and continues with the next
     instance — so one stall now costs ~120s, not the whole pair. Matches the
     standard protocol (seeds `[1,3,5,7,9]` × 100 factuals = 1000 rows total).
     Checkpointed per pair. **COMPLETE** — full 10/10 pairs, 1000/1000 rows,
     ~6.7h total wall clock (much more predictable than v1's uncapped attempt).
     Output: `nice_spars_tabpfn_unskipped_v2.joblib`.

   **Final numbers — materially revises the earlier small-sample read.** The 5-
   factual diagnostic that suggested these two were simply "safe to un-skip" was
   too optimistic. At full scale (120s cap applied consistently):
   | dataset | completeness | timeout rate |
   |---|---|---|
   | `diabetes_binarized` | 82.2% (411/500) | 16–24 stalls per 100-factual seed (13.5% overall, 135/1000 total across both datasets) |
   | `heloc` | 90.0% (450/500) | 5–15 stalls per seed |

   These are **legitimate, usable results, not `not_run`** — NICE-sparse+TabPFN
   genuinely works on these two datasets most of the time, with a real,
   quantified timeout rate under the common 120s budget, same as any other
   imperfect method/dataset combination. Include them as real completeness
   numbers in E2's tables, not as skipped cells.
   - The other 3 originally-skipped combos (`australian`, `blood_transfusion`,
     `chronic_kidney_disease`) are still not part of either run — already have
     sufficient evidence they don't terminate at all (300s+), no new data needed;
     they get `not_run` with the stated reason directly.
   - Output (once complete): `nice_spars_tabpfn_unskipped_v2.joblib`.
3. **Real-world E5b regeneration: COMPLETE** (job 267328). Full corrected
   `dice_tabpfn_corrected_real.joblib`, 4870/4870 rows. See the major-finding
   note above — DiCE+TabPFN completeness is much higher than the buggy numbers
   suggested on most datasets; `ilpd` is the standout hard case.
4. **Merge un-skipped NICE-sparse rows into `output_realdata.joblib`: COMPLETE.**
   Unlike the E5b merge (drop-and-replace of contaminated rows with matching keys),
   this was a pure addition — 0 existing `nice_spars` rows for these 8
   `(model, dataset)` combos before merging, confirmed. Added 4000 rows (1000 from
   `nice_spars_tabpfn_unskipped_v2.joblib`, 3000 from
   `nice_spars_rf_xgb_unskipped.joblib`): 91900 → 95900 total rows. Backfilled the
   static metadata columns that are correctly recoverable from sibling rows
   (`n_cols_model`, `n_rows`, `clf_acc`, `clf_auc`) — verified no gaps after
   backfill. Left genuinely-uncomputed columns (`p_f`, `p_cf`, `margin_cf`,
   `flip_ok`, `rep`, `changed_idx`, `x_f`, `x_cf`, `diff`) as NaN/None rather than
   fabricating them — same gap and same reasoning as the E5b merge (these worker
   scripts never stored the raw vectors needed to derive them). Post-merge
   verification confirmed all 8 combos present with exactly the completeness
   numbers found during characterization (e.g. `TabPFN`/`diabetes_binarized`
   82.2%, `XGB`/`heloc` 99.8%).

5. **Merge corrected DiCE+TabPFN rows into the master joblib caches: COMPLETE.**
   User confirmed backups were made first. Dropped the 4870 (real)/3000 (mock)
   contaminated `dice`+`TabPFN` rows from `output_realdata.joblib`/
   `output_mockdata.joblib`, appended the corrected ones. Verified: row counts
   unchanged (91900/60000), key sets matched exactly before merging (zero
   mismatches either direction), and post-merge completeness-by-dataset numbers
   read directly from the merged files match the standalone correction files
   exactly. Other methods/models (spot-checked `dice`+LR, `pace`+TabPFN, both
   4870 rows) untouched.
   **Known gap**: the corrected rows are missing `x_f`/`x_cf`/`diff` (raw
   vectors) due to a bug in `_e5_lib.py`'s extraction — `include_vectors` was
   accepted as a parameter but the code that actually populates those columns
   was dropped when the function was trimmed for the standalone regen scripts.
   Only affects qualitative-example plotting for `dice`+TabPFN rows specifically
   — all quantitative metrics (`l0`, `l2`, `status`, `ok`, `time_s`) are intact.
   If a specific qualitative example is needed later, regenerate just that one
   `(dataset, seed, idx)` rather than rerunning the full sweep.
6. **E2's full status-normalization script** — not yet built; this log's NICE-sparse
   findings and the decided timeout value are direct inputs to it.

---

## A1/A2 — Proposal-mechanism and hyperparameter ablations (Workstream 5, COMPLETE)

**Shared infrastructure.** `_a1a2_lib.py` extends the `_e5_lib.py` extracted pipeline
(same `prepare_dataset`/`fit_models`/`proba_fn`/`stable_int_seed` used by
`run_benchmark`, so ablation factuals overlap with the main benchmark's evaluated
instances for a given seed) with two ablation-only wrappers:
`guided_cf_build_cache_ablation` (forwards `gamma` into `stageA_build`, previously
not exposed by any harness) and `pace_with_params_ablation` (forwards `order` into
`stageC_select_best`). One additive method-code change was needed in
`pfn_cf_guided_onehot.py`: `stageC_select_best` gained an `order: str = "l0_l2"`
parameter (alternative `"l2_l0"`), changing only which of ℓ0/ℓ2 is the primary
`np.lexsort` key; the default reproduces the exact prior sort order, verified with
a direct before/after comparison on real candidate sets before trusting it in the
full run. `frac_anchor_mix`/`frac_boundary_mix`/`frac_guided_noise` (π_a/π_b/π_n)
needed no method-code change — `stageB_generate` already renormalizes automatically
when one is zeroed (`pfn_cf_guided_onehot.py:606-610`).

### A1 — proposal-mechanism ablation

**Scope, decided with the user up front:** full cross (all datasets × all 4
classifiers) for A1, since the mechanism-importance claim needs to span regimes;
A2 restricted to 2 cells since it's a single-axis default-justification exercise.
Seeds `[1,3,5]` (subset of the standard `[1,3,5,7,9]`), 30 factuals/cell (first 30
of the same per-seed `rng.choice(n_test, size=100)` draw `run_benchmark` uses, so
these are a reproducible subset of already-evaluated instances).

**Initial run** (`a1_proposal_mechanism_ablation.py`/`.sbatch`, job 306683 after a
pilot on job 306679; a100 partition, briefly stuck behind a maintenance
reservation): 4 datasets (`australian`, `blood_transfusion`, `breast_cancer`,
`credit-g`) × 4 classifiers × 3 seeds × 19 configs (baseline + 3 leave-one-out +
15-point mixture-simplex grid over `{0,0.25,0.5,0.75,1.0}`) × 30 factuals =
27,360 rows, 2h22m wall clock. **COMPLETE**, no errors, exact expected row count.
Output: `a1_mixture_ablation.joblib`.

**Generalizability check that triggered a follow-up run.** Computing which
specific instances succeed in the full baseline but fail once one mechanism is
removed ("essential instances" — a mechanism's *sole* route to a flip) showed the
signal was extremely concentrated: 87.5% of noise's essential instances (14/16)
and 69% of anchor's (74/108) came from `breast_cancer` alone — effectively N≈1
for the "boundary/noise matter" claim. Decided with the user to add two more
"hard, high-dimensional, donor-sparse" datasets rather than accept this:
**HELOC** (real, already flagged elsewhere in this log as a hard TabPFN cell) and
the **synthetic d=80 set** (`mc_f80_red40_inf24_seed787359109`, regenerated via
the exact `generate_make_classification_suite` call already used in
`completness.ipynb`'s Appendix B — `n_datasets=30, n_samples=1000,
feature_grid=(5,10,20,40,80), redundancy_grid=(0,0.25,0.5,0.75),
informative_frac=0.3, class_sep=1.2, flip_y=0.1, random_state=42` — so it's the
identical dataset instance, not a resample). `a1_extra_datasets_ablation.py`
imports `a1_proposal_mechanism_ablation`'s `CONFIGS`/`FIXED`/`MODELS`/`SEEDS`
directly (not copy-pasted) to prevent drift. Pilot (job 308306) then full run
(job 308307, 2h19m): 2 datasets × 4 classifiers × 3 seeds × 19 configs × 30
factuals = 13,680 rows, **COMPLETE**, no errors. Merged into
`a1_mixture_ablation.joblib` (pure addition, zero key overlap verified; pre-merge
backup kept as `a1_mixture_ablation.joblib.pre_extra_datasets_merge`) — now
41,040 rows, 6 datasets.

**Result: the follow-up datasets substantially strengthened, not weakened, the
case for keeping all three mechanisms.** Essential-instance counts (sole route
to a flip) across the full 6-dataset, 2160-instance set: anchor 145 (6.7%),
boundary 46 (2.1%, up from 15 — 3×), noise 44 (2.0%, up from 16 — 2.75×).
`breast_cancer`'s share of each dropped to 51%/22%/27% respectively. Critically,
the synthetic d=80 set is boundary's largest single contributor (19/46) and
noise's largest by far (21/44, 48%) — exactly the donor-sparse regime their
design rationale predicts, now demonstrated rather than asserted from one
dataset. Per-cell leave-one-out verdict (24 cells): 12 redundant (no mechanism
necessary, spread <2/90 instances), 12 with real signal (anchor wins 11,
boundary wins 1 — `mc_f80.../LR` — noise never wins a per-cell verdict despite
having real essential-instance value spread thinly across cells).

**Candidate-yield analysis (not in the original A1 spec, added because
completeness alone hides a real cost).** Removing a mechanism has no measurable
runtime effect (median |Δt| ≤ 6.6ms against 0.16–0.9s typical runtimes — Stage C
scores the whole pool in one batched call regardless of composition, confirming
E4's mechanism from a different angle) but a real, consistently-signed effect on
**candidate yield** (% of the M=2000 candidates that actually flip): removing
anchor costs yield in essentially every one of the 24 cells (median −1.6 points,
up to −14.7 on `blood_transfusion`/LR); removing boundary or noise instead
*raises* yield slightly (median +0.5, +0.6), since their freed budget share goes
to the remaining, typically more efficient mechanisms. Visualized in
`a1_yield_penalty_figure.pdf`.

**Pure single-mechanism comparison (also not in the original spec).** The
mixture-simplex grid's three "pure corners" (`(1,0,0)`, `(0,1,0)`, `(0,0,1)`,
each at the full M=2000 budget, no renormalization confound) give a cleaner
three-way comparison than leave-one-out. Median across all 24 cells: anchor
100% completeness / ℓ0=1 / yield 13.1%; boundary 97.8% / ℓ0=1.25 / yield 8.3%;
noise 37.2% / ℓ0=2 / yield 0.9% (noise's ℓ0/ℓ2 numbers are a small-n artifact of
its low completeness — computed over a handful of self-selected-easiest
successes, not a fair comparison to anchor/boundary's 60+ successes; flagged
explicitly in `ablation_analysis.ipynb` wherever cited). On `breast_cancer`
specifically, boundary-only sometimes selects a *closer* point than anchor-only
at comparable ℓ0 (e.g. RF: ℓ0=5/ℓ2=3.05 vs. ℓ0=5/ℓ2=3.15), at a 13.3-point
completeness cost (68.9% vs. 82.2%) — anchor is not simply better, it is
cheaper. Figures: `a1_mixture_response_surface.pdf`, `a1_pure_mechanism_comparison.pdf`,
`a1_essential_instances_by_dataset.pdf`.

### A2 — hyperparameter ablation

`a2_hyperparameter_ablation.py`/`.sbatch`: 24 configs across the 7 axes in the
original spec (M, s, σ_base, α_range, γ, surrogate, lex_order — each axis
including its own default point), on `breast_cancer`/TabPFN (headroom cell) and
`credit-g`/RF (near-ceiling robustness check) × 3 seeds × 30 factuals = 4,320
rows.

**Bug found and fixed mid-run.** The first pass (job 306684, 35min, COMPLETE)
seeded `lex_order`'s two conditions (`l0_l2` vs `l2_l0`) independently, so they
scored *different* candidate pools — a spurious completeness delta appeared
(63.3% vs. 75.6% on `breast_cancer`/TabPFN) even though re-ranking an
already-fixed pool cannot change whether any candidate flips. Root cause:
`order` was folded into the per-config random-seed string alongside every other
axis's value, which is correct for axes that genuinely change what gets
generated (M, s, σ, α, γ, surrogate) but wrong for `lex_order`, which only
changes selection. Fixed by giving `lex_order` a shared seed (independent of
which order); verified with a standalone synthetic-data check that `n_flip` is
now identical between orders for the same seed before re-running. Rerun (job
306758, 35min, COMPLETE) confirmed the fix: completeness is now bit-for-bit
identical between orders (60/90 both, `breast_cancer`/TabPFN; 89/90 both,
`credit-g`/RF) with a real ℓ0/ℓ2 selection difference (ℓ0: 6→7, ℓ2: 3.71→2.92)
instead. Pre-fix file kept as `a2_hyperparameter_ablation.joblib.pre_lexorder_fix`
for reference only — do not cite it.

**Results.** `s` (max changed features) is the dominant, cheap lever on the hard
cell: 14.4%→100% completeness across `{2,4,8,16}`, only +8% runtime for the
`8→16` step. `M` (candidate budget) is a real but far costlier lever: +35.6
points completeness across `{500,...,10000}`, +374% runtime for `2000→10000`
(M directly sets Stage C's batch size). Per unit runtime, `s` is ~44× more
efficient than `M` at recovering completeness (319.8 vs. 7.3 points of
completeness per 100% runtime increase — computed and double-checked directly
from the joblib before citing). The other four axes (σ_base, γ, surrogate,
α_range) move completeness by ≤5 points around the default on the hard cell,
and by ~0 on the easy cell — i.e. the Table 1 defaults are not fragile.
`α_min` widening (`0.5→0.25→0.0`) *lowers* completeness monotonically
(70.0%→65.6%→56.7%), counter to the naive "more exploration helps" intuition —
more weight on the factual itself means candidates move less. Guidance
surrogate is robust in the direction that matters for the paper's efficiency
argument: RF or no importance guidance at all (`γ=0`, uniform unit weights)
costs nothing relative to the default LR surrogate (both ≥ LR by 2.2 points) —
the cheap LR choice isn't leaving performance on the table.

### A3 — write-up

Drafted directly as LaTeX (not built as a script) in three pieces: an opening
motivation paragraph, the "Proposal-mechanism ablation" section (A1 results +
the essential-instance table + the yield-penalty figure), the "Hyperparameter
ablation" section (A2 results + a 7-row per-axis effect-size table), and a short
closing "Are the defaults justified?" synthesis. Every number in all three was
re-derived directly from the joblib files immediately before writing the LaTeX,
not recalled from earlier conversation turns — this caught two real errors
before they reached the manuscript: the A1 per-cell leave-one-out verdict count
was originally mis-stated as 6/24 (actual: 11/24), and the anchor-vs-boundary
completeness gap on `breast_cancer` was first guessed at "15–20 points" before
being replaced with the exact matched RF example (13.3 points, 68.9% vs. 82.2%).
**Deviation from spec:** A3 asked for one merged deltas-vs-default table; two
focused tables (A1's essential-instances-by-dataset, A2's per-axis effect-size)
read more clearly than one combined table would have, so that's what got
written up. The literal combined table the spec asked for still exists as
`ablation_summary_a1a2.csv` (97 rows) for anyone who wants a different view.
Figure color coding (`a1_yield_penalty_figure.pdf`) was matched to
`cfprop_plots.ipynb`'s existing `PROPOSAL_COLORS` (`anchor_mix`=`#56B4E9`,
`boundary_mix`=`#F0E442`, `guided_noise`=`#000000`, cell 38 — the source of
Fig. 9) rather than an independent palette, for consistency with the mechanism
coloring already established elsewhere in the paper.

**Remaining step:** the three LaTeX blocks (opening paragraph, A1 section, A2 +
A3 sections) are drafted and verified but not yet pasted into the manuscript
source, which isn't checked out in this repo — do that, and fill in the
guessed cross-references (`\label{tab:defaults}`, `\label{sec:efficiency}`,
`\label{fig:mechanism-breakdown}`, `\label{sec:mechanisms}`) with the paper's
actual labels.

---

## Workstream 4 — Hierarchical mixed-effects test (S1 extension)

**Problem this addresses.** S1's dataset-level test (3c in `cfprop_plots.ipynb`)
correctly respects the referee's independence objection — it collapses each
dataset to a single median $\Delta\ell_2$ and tests across the $\le 10$
resulting numbers — but that collapse throws away all within-dataset seed/
factual replication, and the result is **0/67 significant cells after
Holm–Bonferroni**, too weak to say anything in the paper. User asked whether
hierarchical modeling could recover power without reintroducing the
pseudoreplication problem.

**Approach.** Added `hierarchical_l2_test`/`combine_hierarchical_results` to
`cfprop_plots.ipynb` (new section "3c′", right after 3c), as a **comparison
against S1, not a replacement**, so both are visible side by side before
committing to one as the headline test. Fits a random-intercept mixed model,
$\Delta\ell_2 \sim 1 + (1 \mid \text{dataset})$, per (baseline, classifier,
$\ell_0$) cell using **every matched instance** (not just 10 per-dataset
medians) via `statsmodels.formula.api.mixedlm`. Because there are only ~10
datasets, a mixed model's own asymptotic p-value/SE for the fixed effect is
known to run anti-conservative in this "few clusters" regime, so significance
is instead based on a **dataset-level cluster bootstrap** (resample *which*
datasets contribute, with replacement, keeping every instance from each
sampled dataset, 2000 resamples) — the standard remedy for few-cluster
inference. Same family-wise correction as S2 (Holm–Bonferroni across the full
baseline $\times$ classifier $\times$ $\ell_0$ family).

**Bug found and fixed while verifying against real data.** The first version
reported the mixed model's own fixed-effect estimate (`mdf.fe_params`) as the
headline effect size. Spot-checking `nice_spars`/XGB/$\ell_0=6$ by hand
(3 datasets contributing 1/13/4 instances — `australian`, `breast_cancer`,
`heloc`) found the true pooled mean $\Delta\ell_2 = +1.271$ (all 18 instances
positive), but the mixed model's fixed effect reported $\approx 0$ — a
degenerate REML solution in this small/unbalanced-cluster case, not a data
issue (confirmed by direct computation from `output_realdata.joblib`). The
cluster-bootstrap significance call was unaffected (it's computed directly
from the raw resampled data, not the model fit), but the reported effect size
would have been actively misleading. Fixed by making `mean_delta_l2` always
the plain pooled mean (immune to model convergence), keeping the mixed
model's estimate as a separate diagnostic column (`mean_delta_l2_lmm`), and
adding an `lmm_unstable` flag (true when the mixed-model point estimate falls
outside its own cluster-bootstrap CI). Re-verified: 17/67 testable cells are
flagged `lmm_unstable` on the real data — expected in this regime, and
correctly does not affect `mean_delta_l2` or `p_boot` for any of them.
Verified end-to-end by executing an isolated copy of cells 0–28 with
`jupyter nbconvert --execute` (not saved into the tracked notebook — the
inserted cells 3c′ in `cfprop_plots.ipynb` are unexecuted as committed; rerun
them to populate outputs).

**Result on `output_realdata.joblib`.** Hierarchical test: **39/67**
significant after Holm–Bonferroni, vs. S1's **0/67** — same correction, same
underlying data, the only difference is not discarding within-dataset
replication before testing. Mean ICC across testable cells: 0.146 (the share
of variance that is between-dataset — roughly how much the naive per-instance
tests in 3a/3b were overstating power by, for reference). Breakdown by
baseline: DiCE 21/23 significant (strong, consistent, all negative — PACE
closer, every classifier, $\ell_0$ 1–7), NICE-sparse 13/32 (a genuine
sparsity-dependent crossover — PACE significantly closer at $\ell_0=1$ for
all four classifiers, reverses to NICE-sparse significantly closer at
$\ell_0 \ge 6$ for RF/XGB/TabPFN), MCCE 5/21 (mixed direction, and mostly
underpowered — 13/21 testable MCCE cells have matched instances from only
1–2 datasets, because MCCE picks lowest-$\ell_0$-available rather than
targeting PACE's $\ell_0$, so it rarely matches exactly). Pareto dominance
(descriptive, unaffected by any of the above): PACE dominates DiCE in 92.3%
of comparable pairs, MCCE in 96.6%, NICE-sparse in 75.8% — consistent with
the significance pattern (weakest dominance rate is NICE-sparse, where the
high-$\ell_0$ reversal lives).

**Grounding script:** `ground_s5.py` (not checked into the repo — throwaway,
re-execs the relevant notebook cells standalone to print/export full-precision
tables; rerun 3c′ in the notebook directly to reproduce). Full result tables
were exported during grounding but not committed (scratch CSVs); the numbers
above and in `draft_S5_significance_rewrite.md` are the citable ones.

**Write-up: DONE — pasted into `pace.tex` directly** (found checked out in the
repo root after all, contrary to earlier notes in this log claiming it
wasn't). User confirmed: hierarchical test is now the headline; S1's literal
dataset-median spec kept as the sensitivity check. Edited every S1–S5-tagged
`\revitem`/`\revcut`/`\revnote` location that pace.tex's own revision-tracking
markup (`\ifrevshow`-gated) flags for this workstream: the abstract
contribution bullet (~l.332), the S1/S2/S3 revitem's resolution note
(~l.1201, Section~\ref{sec:exp}), the F3/S4 revitem's resolution note plus the
TabPFN real-data significance paragraph itself (~l.1406, Section~\ref{sec:real}),
the Pareto-dominance paragraph (added verified pooled dominance rates,
~l.1469), and both Discussion `\revcut{S5}{...}` blocks plus the PFN paragraph's
"Wilcoxon and Pareto dominance criteria" sentence (~l.2173, l.2203, l.2245,
Section~\ref{sec:discussion}). `\revcut`/`\revitem`/`\revnote` bodies were
rewritten in place (kept, not deleted) per the file's own convention — see
`pace.tex:88-108` for what each macro does; with `\revshowtrue` (current
setting) they render as highlighted inline notes, with `\revshowfalse` (clean
build) `\revcut` silently emits only the corrected text and `\revitem`/
`\revnote` disappear entirely, so the manuscript is submission-ready under
either setting. One out-of-scope fix flagged rather than made: the original
TabPFN paragraph's "PACE is even against NICE(base) with a 1--1 split"
sentence was dropped (not softened) because NICE(base) is excluded from the
real-data baseline set entirely per `draft_nice_base_vs_sparse_note.md` — a
pre-existing inconsistency unrelated to S1–S5, noted inline via `\revnote`
rather than silently fixed.

**Remaining steps:** (1) run 3c′ in `cfprop_plots.ipynb` and save outputs so
the notebook itself shows the numbers, not just this log; (2) the
Table~\ref{tab:cond_l2_diff}/Table~\ref{tab:pareto_dominance} appendix data
tables (S2's "add a $p_{\mathrm{adj}}$ column" requirement) were **not**
regenerated in this pass — they're large auto-generated LaTeX blocks pasted
from `export_cond_l2_diff_table`/similar, out of scope for a prose-only S5
pass; needs its own regeneration step analogous to A3's table exports; (3)
~~Section~\ref{sec:synthetic}'s mock/synthetic-data Wilcoxon paragraph left
untouched~~ **DONE, see addendum below**; (4) consider whether S3's effect
size (previously rank-biserial correlation, computed on S1's 10 dataset
medians) should be replaced or supplemented by the hierarchical test's mean
$\Delta\ell_2$ + bootstrap CI, which is now the more informative of the two
given S1's power problem — not decided or acted on.

### Mock/synthetic-data addendum

User pointed out the hierarchical test needed to be run against the synthetic
suite too, not just real data — S1's `\revitem` in `pace.tex` always scoped
the statistics rework to all of Section~\ref{sec:exp} (synthetic + real), and
the earlier pass here only covered real.

**Setup.** `output_mockdata.joblib`: 30 synthetic datasets (an order of
magnitude more than real's 10), 5 methods (`nice_base` included here, unlike
real, where it's excluded per `draft_nice_base_vs_sparse_note.md`). Ran
`hierarchical_l2_test`/`combine_hierarchical_results` (same code, no changes
needed — the functions are already `data_in`-agnostic) via `ground_mock.py`
(throwaway, not committed, mirrors `ground_s5.py`).

**First finding — confirms an existing qualitative claim quantitatively.**
Only `nice_spars` produces any testable cell at all: 0 rows for DiCE, MCCE,
and NICE(base) at every (classifier, $\ell_0$) — none of them reach the
$n\ge5$ matched-pairs threshold on any cell. This is exactly what the
manuscript's existing prose already says qualitatively ("matched pairs at
fixed $\ell_0$ are therefore observed only for NICE" — mock datasets have no
immutables/categoricals, so non-PACE methods rarely land on PACE's exact
$\ell_0$), now confirmed directly from the hierarchical test's own instance
counts rather than assumed.

**Result for `nice_spars`.** Hierarchical: **20/29** significant after
Holm–Bonferroni (mean ICC 0.249, 6/29 cells `lmm_unstable`-flagged — same
diagnostic pattern as real data, at a similar rate). S1's literal
dataset-median test: 4/13 significant — still weaker than the hierarchical
version, though less dramatically than on real data (30 datasets gives it
more room than real's 10, but it's still both lower-count *and*
lower-hit-rate: 13 testable cells vs. hierarchical's 29, because collapsing
to one median per dataset also collapses away cells where a dataset has too
few *matched* instances to compute a stable per-dataset median at all, even
before the signed-rank test itself runs).

**The headline pattern replicates independently on a different dataset
family.** Same low-$\ell_0$-PACE-wins / high-$\ell_0$-NICE-sparse-wins
crossover found on real data: PACE significantly closer at $\ell_0=1$ across
all four classifiers ($p_{\mathrm{adj}}<0.001$, $n\ge500$, all 30 datasets
represented every time), NICE(sparse) significantly closer at $\ell_0\ge3$ for
LR/TabPFN/XGB and at $\ell_0\in\{2,4,6\}$ for RF specifically. This is a
genuine cross-validation of the real-data finding on an independent
(synthetic, no immutability/categorical structure) dataset suite — worth
citing as evidence the crossover is a property of the methods, not an
artifact of the real-world dataset selection.

**One correction to the existing manuscript text.** The paragraph in
Section~\ref{sec:synthetic} said "for $\ell_0=2$ the difference is not
statistically significant" as a blanket claim. Per-classifier, that's true
for LR/TabPFN/XGB but **not** RF, where $\ell_0=2$ *is* significant favouring
NICE(sparse) ($p_{\mathrm{adj}}=0.044$, $n=91$, 24/30 datasets). Fixed in
`pace.tex` rather than left as a stale claim.

**Text status: pasted into the manuscript**, same pattern as the real-data
pass — added a `\revitem{S1, S2, S3}` resolution block in
Section~\ref{sec:synthetic} with the numbers above, corrected the
per-classifier $\ell_0=2$ claim, and updated the S1/S2/S3 resolution note in
Section~\ref{sec:exp}'s intro (previously said synthetic was out of scope)
and the abstract bullet's `\revnote` to point here instead of flagging it as
open.

---

## E4 — TabPFN cost instrumentation (COMPLETE)

**Goal.** Demonstrate, not just assert, that PACE's efficiency comes from a
single batched target-model pass while baselines query the target
iteratively — by actually counting target-model `predict_proba` calls and
rows evaluated per method, not just citing the (already code-verified)
mechanism.

**Why a new instrumentation layer, not a run_benchmark edit.** `_e5_lib.py`'s
`run_benchmark` (the E5b-patched extract of `tabpfn_cf5.ipynb` cell 15) turned
out to have two of its five method branches dead: `nice_base`/`nice_spars`/
`mcce` are gated on `nice_base_obj`/`nice_spars_obj`/`mcce_obj`, which are
initialized to `None` and never built in that extraction (the build blocks
only exist in the notebook's fuller version of the cell — dropped because the
E5b task only needed `"dice"` to work). Separately, `run_benchmark`'s `"pace"`
branch calls `guided_cf_build_cache`/`guided_cf_one_repeated`, which are
**not defined anywhere importable** — they exist only as notebook cell source
in `tabpfn_cf5.ipynb`/`completness.ipynb`, never extracted to a `.py` (this
is also why `_a1a2_lib.py` wrote its own `guided_cf_build_cache_ablation`/
`pace_with_params_ablation` instead of reusing `_e5_lib`'s — those names
never existed there to reuse). Confirmed by direct import: `'guided_cf_build_cache'
in dir(_e5_lib)` is `False`, and every existing driver script that imports
`_e5_lib` only ever passes `methods=("dice",)`, so the gap was never hit
before. New code: **`_e4_lib.py`** — `from _e5_lib import *` for everything
that does work (`prepare_dataset`, `fit_models`, `fit_models_for_dice`,
`build_nice_explainer`/`nice_cf_one`, `dice_to_standard_out`/`validate_cf`,
`choose_dice_method`, etc.), plus (a) `guided_cf_build_cache`/
`guided_cf_one_repeated`/`GuidedCFCache` reproduced verbatim from
`tabpfn_cf5.ipynb`'s cell immediately before `run_benchmark` (the one latent
bug fixed rather than preserved: `feature_info`'s default was the literal
runtime expression `FeatureInfo | None`, a `UnionType` object, not `None` —
never hit because every call site passes it explicitly, but written correctly
here since this is new code), and (b) `run_benchmark_instrumented`, which
restores the `nice_base`/`nice_spars`/`mcce` build blocks from the notebook's
fuller cell (so all five methods actually run) and adds the call counter.

**Instrumentation mechanism.** `count_predict_proba(obj)` (`_e4_lib.py`) is a
context manager that monkeypatches `obj.predict_proba` to a counting wrapper
for the duration of the `with` block, then restores the exact prior state
(instance-dict value or its absence) — verified safe against dice-ml's
`base_model.py`, which calls `self.model.predict_proba(...)` fresh on every
invocation (not a cached bound reference from `Dice(...)` construction time),
so patching the pipeline object right before `generate_counterfactuals` and
un-patching right after reliably brackets exactly DiCE's own search calls.
Confirmed the same holds for NICE (both the raw-space and OHE-fallback
`predict_fn` closures call `model.predict_proba` by attribute lookup) and
MCCE (`_ModelWrapper.predict` likewise). Patched object per method: the
plain target `model` for `pace`/`nice_base`/`nice_spars`/`mcce`; the DiCE
pipeline `dice_spec['models'][model_name]` for `dice` (a different, cloned
estimator object than `model`, fit on raw-to-preprocessed data — patching
`model` would not have caught DiCE's calls at all). **Scoping choice:** only
the method's own generation/search call is wrapped (`guided_cf_one_repeated`,
`nice_cf_one`, `mcce_cf_one`, `dice_exp.generate_counterfactuals`) — not the
harness-level post-hoc checks that run uniformly after (the common final
`pp(x_cf)` metric recomputation outside the if/elif chain, and, specifically
for DiCE, `dice_to_standard_out`/`validate_cf`'s re-check on the plain
`model`). Those are constant overhead across every method and would only add
noise to a comparison that's about each method's own search cost.

**Validation before the real run.** `e4_pilot_test.py` (blood_transfusion,
LR+RF, 8 factuals, all 5 methods) surfaced one genuine edge case, not a bug:
PACE's target-call count was 3 (not the expected 2) whenever **the target
model is LR** — because PACE's guidance model is always LR, so when the
target is *also* LR, `stageB_generate`'s one guidance-side confidence check
(`pfn_cf_guided_onehot.py:445`, on the guidance model) and Stage C's
target-side batch call happen on the literal same object, and the counter
correctly can't distinguish "guidance" from "target" calls in that case. For
every other target (RF/XGB/TabPFN — the actual "expensive predictor" cases
the paper's efficiency claim is about) the two models are different objects
and the count is exactly 2, confirmed on both the pilot and the full run
(min=max=2 for RF/XGB/TabPFN in every dataset). `e4_dryrun_test.py`
(credit-g, LR+RF, 4 factuals) additionally exercised the categorical/
immutable-feature code paths the numeric blood_transfusion pilot didn't
touch (`nice_base` correctly returns `violates_immutable` on credit-g's
immutable features; MCCE builds and runs cleanly with categorical
immutables) before committing to the GPU run.

**Full run.** `e4_tabpfn_cost_instrumentation.py` / `.sbatch` (job 310026,
18 min wall clock on the `a100` partition — far faster than the throttled
login-node dry run suggested, since the login node is CPU-oversubscribed and
not representative). 4 datasets × 4 classifiers × 5 methods × 15 factuals ×
1 seed = 1185 rows (`e4_tabpfn_cost.joblib`), plus one row per (dataset,
model) recording the one-off PACE cache-build cost (`e4_tabpfn_cost_cache_builds.joblib`).
Datasets, chosen for the same reasons as A1's representative subset plus one
constraint specific to E4/E1b — need a continuous-only set so DiCE runs
under **genetic** (the population-based search E1b's scaling argument is
about), which only `choose_dice_method` selects for names starting `mc_`/
`synthetic`:
- `mc_f80_red40_inf24_seed787359109` — synthetic, 80 continuous features,
  the identical instance A1 used (`generate_make_classification_suite`,
  same call signature) → DiCE runs **genetic** here, giving E1b's requested
  empirical anchor for "GA-style search is call-hungry" without any MOC
  integration.
- `breast_cancer`, `credit-g`, `heloc` — mixed real-world, DiCE runs
  **random** (per `choose_dice_method`); `heloc` is the largest (real
  TabPFN-scaling case), `breast_cancer` the hard TabPFN-incompleteness case
  flagged throughout the revision. `nice_spars`+TabPFN+`heloc` skipped (only
  entry kept from `tabpfn_cf5.ipynb`'s historical skip set relevant to these
  4 datasets — a known hang pre-dating the E2/E3 120s-timeout decision).

**Results (`tabpfn_cost.csv`, `tabpfn_cost_overall.csv`, pooled across the 4
datasets; `e4_analysis.py` regenerates both plus the plot below).**

| model | method | mean target calls/CE | mean target rows evaluated/CE | mean time_s |
|---|---|---:|---:|---:|
| RF/XGB/TabPFN | **pace** | **2.00** | ~1534 | 0.17 / 0.17 / 1.33 |
| LR | pace | 3.00 (guidance==target edge case above) | 1534 | 0.17 |
| any | mcce | ~1.97–2.00 | ~1001 | 0.22–1.27 |
| any | nice_base | 2.75 | 2.75 | 0.004–1.31 |
| RF/XGB/TabPFN | nice_spars | 14.2–17.5 | 163–226 | 0.02 / 0.02 / 3.97 |
| LR | dice (random, real-world sets) | 37.6* | 1820 | 0.26 |
| RF/XGB/TabPFN | dice | 9.4–10.9 | 2574–3558 | 0.20 / 0.68 / 5.18 |

\*LR/DiCE's mean is skewed by one outlier real-world instance at 1743 calls
(median is 6, matching RF/XGB/TabPFN) — DiCE's iterative search occasionally
runs long searching for a flip; median is the more representative summary,
both are in `tabpfn_cost.csv`.

**Confirms the verified mechanism exactly:** PACE = 2 target calls/CE for
every expensive-predictor case (RF/XGB/TabPFN), one of them the single
batched Stage C pass — never varying by dataset or factual, unlike every
baseline. NICE-sparse and DiCE are the call-hungry ones (7–48 calls/CE,
dataset- and instance-dependent), consistent with "iterative baselines call
the target throughout their search."

**Honesty-check findings (per E4 spec's explicit ask — don't just confirm the
existing prose):**
1. **The guidance/Stage-A cost is genuinely negligible**, confirmed
   empirically, not just architecturally: `e4_tabpfn_cost_cache_builds.joblib`
   shows the one-off LR-guidance build costs 8–44 calls and 0.008–0.04s per
   (dataset, model) — a rounding error next to a single TabPFN forward pass
   (0.6–2.7s per PACE call in the per-row data).
2. **Calls → runtime is NOT a universal predictor — it's specifically the
   expensive-predictor story.** Spearman correlation of `target_calls` vs.
   `time_s`, pooled across all 5 methods and 4 datasets, per classifier:
   TabPFN ρ=0.654 (p=3×10⁻³⁶), RF ρ=0.598 (p=2×10⁻³⁰) — strong, matching the
   paper's causal claim — but **LR ρ=−0.204 and XGB ρ=−0.301** (both
   significant, both *negative*). For fast closed-form/optimized-inference
   models, call count does not gate wall-clock; other per-call/per-iteration
   overhead (Python-level looping in the baselines' search loops) dominates
   instead. This is exactly the "state both, not batching alone" outcome the
   spec asked for: the batched-single-pass mechanism is the reason PACE is
   cheap *under expensive predictors specifically* (TabPFN, and to a lesser
   extent RF), not a universal claim across every classifier. Visible directly
   in `e4_runtime_vs_calls.pdf`'s panels: RF/TabPFN show a clear rising
   diagonal across all methods; LR/XGB show two flat, separated clusters
   instead (baselines' calls span an order of magnitude at roughly constant,
   method-specific runtime).

**Plot.** `e4_runtime_vs_calls.pdf` — log-log scatter, one panel per
classifier (LR/RF/XGB/TabPFN), points colored by method (5 of the 8
Okabe-Ito colorblind-safe hues, the same family as `cfprop_plots.ipynb`'s
`PROPOSAL_COLORS`, chosen by that palette's known CVD-separation property
rather than re-validated with the dataviz skill's `validate_palette.js` —
this environment's only available `node` is v12, too old for the script's
ES2021 syntax). Generated by `e4_analysis.py`, which also writes
`tabpfn_cost.csv` (per dataset × model × method) and
`tabpfn_cost_overall.csv` (pooled per model × method — the headline table).

**Feeds forward:** E1b's call-count/latency-projection argument for the
declined MOC benchmark (uses this run's PACE call counts directly, and its
DiCE-genetic-on-synthetic row as the empirical GA-cost anchor); E3's
cost-asymmetry framing (target-model call counts alongside the common
wall-clock budget).

**Remaining step:** write the Section 4 paragraph(s) stating the timer
boundary (Stage A excluded from per-CE timing, symmetric with baselines'
excluded explainer builds — already verified from code, see the action
list's E4 spec) and citing the table/plot above; not yet drafted as LaTeX.

**Update — pasted into the manuscript.** The user shared `pace.tex`, which
turned out to already contain a placeholder `\subsection{Model-call
accounting and scaling}\label{sec:calls}` holding exactly the `\revitem{E4}`
and `\revitem{E1, E1b}` spec notes, immediately adjacent (both items share
one subsection in the paper's actual structure). Wrote the real prose,
`tab:calls_e4` (median calls/time by method × classifier, from
`tabpfn_cost_overall.csv`), and `fig:runtime-vs-calls`
(`e4_runtime_vs_calls.pdf`) directly into that subsection, replacing the
revitem block — see the E1b entry below for the MOC/scaling half pasted
immediately after it in the same edit. Side effect: this subsection's new
`\label{sec:efficiency}` (added alongside the pre-existing `\label{sec:calls}`)
resolves two previously-dangling `\ref{sec:efficiency}` citations elsewhere
in the document (the proposal-mechanism and hyperparameter ablation
sections, lines ~1572/1696 pre-edit) that had no matching label before —
found while checking the new content's cross-references, not something I
went looking for. The timer-boundary paragraph (Stage A excluded from
per-CE timing) is still not explicitly stated in this subsection — it may
already be covered by the existing M6 discussion earlier in Section 4
(`pace.tex`'s guidance/target paragraph, which has its own pending
`\revnote{M6, E4}`); worth checking both aren't needed in both places once
M6 is resolved.

---

## E1b — Scaling analysis: population-based EA vs. PACE under expensive predictors (COMPLETE)

**Purpose.** Replaces the declined MOC benchmark (E1's decision — MOC has no
clean Python/TabPFN drop-in and integrating it faithfully is disproportionate
effort for a method the paper already argues is architecturally mismatched).
Turns "it won't scale" into a quantified argument: how many target-model
calls does a population-based EA (NSGA-II/MIES, as MOC uses) need per
counterfactual, and what would that cost under TabPFN specifically.

**No new run — pure analysis on E4's data.** `e1b_scaling_analysis.py` reads
`e4_tabpfn_cost.joblib` directly. E4's run already included DiCE under
**genetic** on the synthetic dataset (`mc_f80_red40_inf24_seed787359109` —
`choose_dice_method` selects genetic for `mc_`-prefixed names), across all 4
classifiers, giving a real, measured population-based-EA-under-TabPFN
datapoint with zero additional compute — exactly the spec's "optional
empirical anchor... no MOC integration necessary."

**Three things on one axis, kept clearly distinct:**
1. **PACE — measured**, same synthetic dataset (apples-to-apples with #2):
   2 target calls/CE (RF/XGB/TabPFN; 3 for LR-as-target, the E4-documented
   guidance==target edge case), 1.07s/CE under TabPFN.
2. **DiCE-genetic — measured**, same dataset: dice_ml's genetic optimizer
   under **its own defaults** (not MOC's) ran **exactly 16 generations ×
   ~51 population = 819 rows, in 16 calls, for every single one of the 60
   (factual × classifier) combinations checked** (min=max=16 calls,
   100% completeness) — deterministic, not an artifact of a small sample.
   Per-call TabPFN cost: 5.153s / 16 = **0.322s/call**. This is explicitly
   labeled as dice_ml's own hyperparameters, not a MOC stand-in, so it can't
   be mistaken for a MOC result.
3. **MOC — analytical projection**, using MOC's *published* defaults
   (Dandl et al.: population ≈20, generations ≈175 → 20×176 = **3,520
   calls/CE**, matching the action list's own "≈3,500" estimate), latency-
   calibrated against DiCE-genetic's own measured per-call TabPFN cost from
   #2 above (same algorithm family — a population-based EA issuing one
   sequential batched query per generation — same hardware, same codebase;
   the most defensible calibration available short of actually integrating
   MOC). This is clearly marked "projected" everywhere (hatched bar fill in
   the figure, separate `time_basis` column in the CSV) so it's never
   conflated with a measured result.

**Headline numbers (TabPFN, `e1b_scaling_table.csv`):** PACE measured
**1.07s/CE** (2 calls); MOC projected **1,133.6s/CE ≈ 18.9 minutes** (3,520
calls) → **~1,064× slower, projected**. Per-call latency scales with
classifier cost exactly as expected from the calibration (LR 0.0185s/call →
MOC projected 65.1s/CE; RF 0.0582s/call → 204.7s/CE; XGB 0.0216s/call →
76.1s/CE; TabPFN 0.322s/call → 1,133.6s/CE) — the EA's absolute cost tracks
whichever classifier is expensive per-call, same pattern E4 already found
for calls-vs-runtime correlation (strong for TabPFN/RF, weak for LR/XGB).

**"Batching is the lever" — supported directly by the measured numbers, not
asserted.** DiCE-genetic's calls average **51 rows/call**; PACE's Stage C
call scores **~2,000 candidates in one call**. Despite the ~40× difference
in per-call batch size, PACE's overall per-call TabPFN cost (1.065s / 2 ≈
0.53s/call, one of those calls a singleton) is the same order of magnitude
as DiCE-genetic's per-call cost (0.322s/call, at batch≈51) — consistent
with TabPFN's forward-pass cost being dominated by re-encoding the training
context rather than by query-batch size (the mechanism already noted in
`scaling_tests.ipynb`). The EA's problem was never that its individual
queries are expensive — it's that it needs **3,520 of them, sequentially,
because each generation depends on the last**, while PACE pays that
per-call tax exactly twice by generating its whole candidate pool up front
and scoring it in one shot.

**Outputs:** `e1b_scaling_table.csv` (the 3-method × 4-classifier table
above), `e1b_calls_per_ce_bar.pdf` (log-scale grouped bar chart, calls/CE by
classifier; MOC's bars hatched to visually distinguish projected from
measured). Reuses 2 of the 3 Okabe-Ito hues from `e4_analysis.py`'s method
palette (PACE blue, DiCE vermillion) plus a neutral gray for the MOC
projection specifically (a fourth color would be one too many categorical
hues doing double duty with E4's plot, and gray + hatch is the standard
"this is not measured data" treatment).

**Text status: pasted into the manuscript.** Written directly into
`pace.tex`'s `\subsection{Model-call accounting and scaling}` (`sec:calls`),
immediately after the E4 prose/table/figure above and in the same edit —
covers the honest scope, the `tab:moc_scaling` table (PACE measured /
DiCE-gen measured / MOC projected, 3 rows × 4 classifiers), the "batching is
the lever" paragraph tying back to E4's mechanism, and the
`diabetes_binarized` catastrophic-tail datapoint (~8% of instances at
~2000s under DiCE-random+TabPFN, all 5 seeds — cited exactly as the
manuscript's own `\revitem{E1, E1b}` note had asked for, pulled from this
log's E5b entry) as corroborating evidence alongside the analytical
projection. Used the manuscript's actual bib key `dandl2020multi` (not the
`dandl2020moc` placeholder guessed in `draft_E1_moc_scaling_subsection.md`
before `pace.tex` was available) and its real section labels
(`sec:related`, `sec:exp`, `sec:tradeoff`, `sec:stageA_importance`,
`sec:stageB`, `sec:stageC_select`). `draft_E1_moc_scaling_subsection.md` is
now superseded by this edit — kept for its rebuttal-letter paragraph, which
wasn't pasted anywhere (no rebuttal letter file exists in this repo yet).

**Feeds into R1w** (MOC's Related Work positioning, Section 2 — still an
unresolved `\revitem` in `pace.tex` as of this edit): that item should
describe what MOC *is* and how it differs methodologically; this subsection
covers why it isn't *run* here, quantified — the two are complementary, not
overlapping, but worth a final read-through together once R1w is drafted to
confirm no duplication.

---

## Workstream 6 — Figures (F1–F5, COMPLETE)

All five items regenerated/redesigned directly against the current, already
E5b- and E2-corrected `output_realdata.joblib`/`output_mockdata.joblib` (no
new benchmark run — pure plotting-code changes in `cfprop_plots.ipynb`,
executed via `jupyter nbconvert --execute --inplace`). Pasted into `pace.tex`
with the corresponding `\revnote`s marked RESOLVED.

**Correctness fix found while regenerating F1 (real data)**: cell 11's
completeness bar computed `df.groupby(...)["flip_ok"].mean()`, and pandas
`.mean()` silently drops `NaN` rows from the average instead of counting them.
The ~3,800 `nice_spars` rows merged in for E2's un-skip fix
(australian/diabetes_binarized/heloc/blood_transfusion × RF/XGB/TabPFN) have
`status == "ok"` (correct) but `flip_ok` was never backfilled (`NaN`) — so
these real successes were silently excluded from the denominator, undercounting
exactly the completeness that E2's fix was meant to recover (e.g.
`nice_spars`/TabPFN measured 72.5% instead of the correct 76.6%; RF 83.1%
instead of 87.7%; XGB 84.4% instead of 89.1%). Fixed by backfilling only the
missing values — `df["flip_ok"].where(df["flip_ok"].notna(), (df["status"]=="ok").astype(float))`
— rather than replacing `flip_ok` outright: a real, separate finding surfaced
in the process is that `status == "ok"` and `flip_ok == 1` are *not*
interchangeable in general (61 `pace`/TabPFN rows have `status == "ok"` —
candidate generation succeeded — but `flip_ok == 0` — the returned point
didn't actually cross the decision boundary), so a blanket `status == "ok"`
substitution would have quietly introduced a new, different miscount. This
fix changed the headline real-world completeness numbers cited in Section 4
(see F1 below) and, downstream, which baseline shows "the largest drop" under
TabPFN.

**F1 — enlarge the ℓ0–ℓ2 panels of Figs. 2 (mock) and 5 (real). DONE.**
Both figures previously packed a small l0-l2 scatter (all 4 classifiers
overlaid in one panel) next to the flip-rate bar chart in a single `figure*`.
Split into two figures each: the bar chart stays a compact `figure`, and the
scatter becomes a full-width `figure*` with one subplot per classifier
(`{mock,real}_l0_l2_by_model.pdf`, replacing `{mock,real}_flip_l2_l0.pdf`).
New labels `fig:mockdata_l0_l2`/`fig:real_l0_l2`; `fig:mockdata_ce_summary`/
`fig:real_completeness_l0_l2` now refer to the bar-chart-only figure. Also
regenerated against the E5b/E2-corrected data as the existing `\revnote` asked
— this is what surfaced both the `flip_ok` backfill bug above and the
completeness-magnitude finding below.

**Knock-on finding (E5b/E2 text, fixed as a necessary consequence of
regenerating these two figures — not a new Workstream-6 item, flagged for
Workstream 3's own record):** the synthetic-suite prose ("a marked drop in
the completeness of DiCE when applied to TabPFN... even though fully
continuous") is now false on the corrected data — DiCE-genetic measures 99.2%
completeness under TabPFN on the synthetic suite (LR/XGB 100%, RF 99.9%), i.e.
no drop at all; the pre-fix number was entirely the stale-RF-explainer
artifact. On the real-world side, DiCE/TabPFN completeness (pooled) is 88.1%
post-fix — no longer the lowest of the compared baselines; NICE(sparse) is
now the largest drop (76.6%, partly datasets whose nice\_spars/TabPFN cells
were only recently recovered from `not_run` by the E2 timeout fix). Rewrote
both passages in `pace.tex` (the `\revitem{E5b}` block and the real-data
completeness paragraph) with the corrected numbers and a `\revnote{E5b}{RESOLVED...}`
marker, since leaving the old prose next to a regenerated figure that
visibly contradicts it would have been a worse inconsistency than the
original one. The fuller Section 4/E5 rewrite (the genetic-vs-random
narrative, the decoupling argument) is **not** done here — flagged as
still-open Workstream 3 business, not claimed as resolved.

**F2 — fix Fig. 3 (mock forest plot) row labels. DONE.**
The old `mock_cond_l2_diff_AGG.pdf` (generated by a `plot_conditional_wilcoxon_aggregated`
function no longer present in the notebook — stale since May) labelled every
row "PACE − nice\_spars" regardless of nominal baseline, because only
NICE(sparse) ever has a matched-ℓ0 cell in the synthetic setting (DiCE/MCCE/
NICE(base) have zero testable cells, confirmed via the hierarchical test, not
assumed). Rather than relabel a design that had nothing to relabel, regenerated
using the same ℓ0-on-x-axis, per-classifier design already built for F3
(`plot_l0_vs_l2diff_by_model`, applied to mock's own `dataset_stats`) —
`mock_l0_vs_l2diff_by_model.pdf`. Only NICE(sparse) points appear, by
construction; the caption states explicitly why the other three baselines
have nothing plotted, so figure and text now agree without a special-case
label fix. Visually confirmed (only orange/NICE(sparse) markers render, one
legend entry) before pasting.

**F3 — redesign Fig. 6 (real forest plot) with ℓ0 on the x-axis. DONE (code
already existed from the S1 rework — this pass regenerated it against
current data and pasted it into `pace.tex`).** `plot_l0_vs_l2diff_by_model`
(section 3d of the notebook) was already implemented and had produced
`real_l0_vs_l2diff_by_model.pdf` on 2026-08-31, but that predates both the
E5b merge and the NICE-sparse un-skip merge (2026-09-04/05) and had never
actually been pasted into `pace.tex` — the manuscript still pointed at the
old `real_cond_l2_diff_AGG.pdf`. Regenerated against current data and swapped
in, replacing the `\revitem{F3, S4}`'s "still open" note with RESOLVED.

**F4 — redesign Fig. 7 (Pareto dominance), too many overlapping points.
DONE.** The original design jittered all 10 real-world datasets on top of
each other within each (model, baseline) group, using one marker shape per
dataset — but only 8 marker shapes were defined for 10 datasets, so
`ilpd`/`australian` silently shared a circle and `sick`/`blood_transfusion`
silently shared a square (confirmed directly from the marker-assignment code,
not just visually), compounding exactly the overlap R3 flagged. New
`plot_pareto_dominance_small_multiples` function: one panel per dataset,
baseline as the only colour channel needed within a panel. Per the spec's
explicit fallback ("if space-limited, keep a representative subset... move
the rest to the appendix"): main text keeps 4 datasets chosen to span the
cases the prose discusses (`breast_cancer` — where PACE does *not* dominate
NICE(sparse); `diabetes_binarized`/`heloc` — the two datasets recovered by
the E2 un-skip fix; `credit-g` — already used for the qualitative example),
full 10-dataset grid moved to a new appendix figure
(`fig:real_pareto_appendix`, Appendix~\ref{sec:benchmark_results}).
R3's optional "sparsity-vs-proximity intuition scatter" suggestion is
**not** added — flagged in the caption as a possible future addition rather
than silently dropped. Verified the pooled dominance rates quoted in the
surrounding prose (92.3%/96.6%/75.8% vs. DiCE/MCCE/NICE(sparse)) are
unchanged by the E5b/E2 data corrections before leaving them as-is.

**F5 — general polish. DONE (scoped).** Colour palette was already
Okabe-Ito (colour-blind-safe) throughout — no change needed. Fixed the
broken-subscript-in-PDF-text-layer complaint (X3): matplotlib's default
`pdf.fonttype`/`ps.fonttype` is 3 (Type 3), which embeds glyphs with no
`ToUnicode` CMap and is the standard cause of exactly this symptom (subscripts
like ℓ₀/ℓ₂ becoming garbled or unselectable when text is extracted from the
PDF). Set both to 42 (TrueType) globally in the notebook's first cell — a
one-line fix applying to every figure regenerated from this notebook going
forward. Added a one-sentence takeaway to the two plainest captions touched
by F1 (mock flip-rate and mock ℓ0-ℓ2); did not do a full caption pass over
figures untouched by F1–F4 (out of scope for this pass — the ablation and
E4/E1b figures already state takeaways from their own workstreams).

**Known gap surfaced, not caused, by this pass:** re-executing the full
notebook (needed to regenerate the figures above against current data)
exposed that the qualitative-example cells (`inspect_rep_cell`/`cf_table_cell`/
`cf_dot_cell`, feeding the credit-g/breast-cancer qualitative-example
figures) now fail with `IndexError` decoding `x_f`/`x_cf` for one factual —
this is the already-documented "Known gap" from the E5b merge (corrected DiCE
rows are missing raw `x_f`/`x_cf`/`diff` vectors; see the E5b section above).
These cells previously had stale successful outputs from before that merge.
Not fixed here (out of Workstream 6's scope, and `real_cf_example_credit_g.pdf`
is not regenerated by this notebook at all — it predates it); flagging so it
isn't mistaken for a regression introduced by this pass.

**Files regenerated:** `mock_flip_rate.pdf`, `mock_l0_l2_by_model.pdf`,
`mock_l0_vs_l2diff_by_model.pdf`, `real_flip_rate.pdf`,
`real_l0_l2_by_model.pdf`, `real_l0_vs_l2diff_by_model.pdf`,
`real_pareto_small_multiples_main.pdf`, `real_pareto_small_multiples_all.pdf`.
`real_pareto.pdf`/`mock_cond_l2_diff_AGG.pdf`/`{mock,real}_flip_l2_l0.pdf` are
no longer referenced from `pace.tex` but left on disk rather than deleted.

---

## L1/L2 — Broadened relaxation ladder, corrected baseline-$\ell_0$ comparison, and a NICE-specific float32 precision bug (COMPLETE)

**L1's decisive check, done properly.** L1 asks: on the Breast-Cancer/TabPFN
instances PACE's default config fails and relaxation (`max_changed_features`
$8\to20$, config `C_more_feats`) recovers, is PACE-relaxed's median $\ell_0$
actually lower than each baseline's, on the *same* instances? Computed
directly from `output_realdata.joblib`, no new runs needed for the baseline
side (they'd already been benchmarked on Breast Cancer). Result is mixed, not
a clean win: PACE-relaxed (median $\ell_0=11$) beats NICE(base) (30.0) and
MCCE (29.0) clearly, but loses to NICE(sparse) (8.0) and is roughly tied with
DiCE (median 8.0 among the 66% of instances DiCE solves at all — its
completeness on this hard subset is worse than its own 87.4% overall rate on
Breast Cancer, i.e. its failures concentrate disproportionately on exactly
these instances).

**L2: broadened to 3 more datasets, using the exact same harness.** Reused
`completness.ipynb`'s relaxation-ladder pipeline (5 seeds, 6 configs,
`stageA`/`stageB`/`stageC` calls identical to the main benchmark), ported to
a parametrized standalone script (`l2_relaxation_ladder.py <dataset>
<model>`) so it can run unattended via SLURM rather than only interactively.
Picked Australian (36 failures, no immutable features — same structural
regime as Breast Cancer, cheapest to run) and ILPD (45 failures, *with*
immutable features — age/sex — testing a regime Breast Cancer can't) per the
user's request to prioritize structural diversity over raw failure count;
added HELOC (56 failures, largest of Table 4's remaining incomplete cells)
as a third. Ran all three in parallel as separate SLURM jobs (independent,
no shared state, no reason to serialize) — a real efficiency win over the
serial-pairs design forced on the earlier E2 diagnostic (that one had a
sequential per-worker constraint the relaxation ladder doesn't share).

**A real bug, caught before wasting GPU time on it twice.** `ilpd` and
`heloc` both crashed immediately with `TypeError: safe_immutable_cat() takes
2 positional arguments but 3 were given` — a **previously-latent bug already
present in `completness.ipynb`** (traced, not introduced by the port): its
`prepare_for_completeness` calls
`safe_immutable_cat(cat_cols, cat_groups, immutables.cat)`, but
`preprocessing.py`'s actual signature is `safe_immutable_cat(cat_cols,
immutable_cat_names)` — no `cat_groups` argument at all. Never triggered
before because Breast Cancer and Australian both have zero immutable
features, so `prepare_for_completeness` takes the `feature_info = None`
branch and never reaches this call; `ilpd`/`heloc` are the first cases in
this entire project to exercise it. Verified the fix (drop the extra arg)
against both datasets with a cheap CPU-only check before resubmitting the
GPU jobs, rather than finding out via a second failed job.

**Rebuilt Breast Cancer through the same script too**, since the original
`completness.ipynb` run never saved its `sweep_df` to disk (only figures) —
needed a real joblib to build a cross-dataset plot from a consistent source.
Numbers matched the original exactly (median $\ell_0=11$, 147/147 recovered),
confirming the port is faithful.

**Full recovery-rate / cost summary** (config `C_more_feats`, TabPFN):

| dataset | recovered/failed | PACE-relaxed median $\ell_0$ | sparser-and-comparably-complete baseline(s) |
|---|---|---|---|
| Breast Cancer | 147/147 (100%) | 11 | NICE(sparse) only (8.0, 99% complete) |
| Australian | 36/36 (100%) | 8 | DiCE (2.00, 100%), MCCE (6.00, 100%) |
| ILPD | 24/45 (53%) | 8 | none — DiCE 0% complete, NICE <25%, MCCE 71% at $\ell_0=7$ |
| HELOC | 51/56 (91%) | 11 | DiCE (5.00, 92%), NICE(sparse) (5.00, 76%) |

**The aggregate finding is a genuine correction to the paper's original
"still sparser" framing.** Across 4 datasets, PACE-relaxed is unambiguously
the sparsest-while-complete option in only 1 of 4 (ILPD — and there for a
different reason: every baseline's completeness collapses, not because PACE
is sparse). It loses on sparsity outright in 2 of 4 (Australian, HELOC). The
honest generalization is **"relaxation reliably recovers completeness; it
does not reliably preserve a sparsity advantage over baselines"** — a
materially different, more defensible claim than "PACE remains sparsest,"
and one that required the broadened, multi-dataset check L1/L2 asked for to
even see.

**A correction to my own comparison code, caught before it reached anyone.**
The baseline-comparison print statement in `l2_relaxation_ladder.py`
computed the denominator as the method's *full 500-query protocol count*
(`(other_raw["method"]==m).sum()`) instead of the actual comparison-set size
(the number of PACE-C\_more\_feats-recovered instances, e.g. 24 for ILPD,
not 500) — every baseline's printed fraction ("X/500") was wrong,
understating every baseline's real performance on the instances that
matter. Recomputed correctly (`n_target = len(target_pairs_set)`) before
citing any number above; the underlying `l2_sweep_*.joblib` files were
never wrong, only the print statement's denominator was.

### Completeness-vs-sparsity figure (`l2_completeness_vs_sparsity.pdf`)

User asked for a direct visual of the L1/L2 table above — one point per
(dataset, method) at (median $\ell_0$, completeness %) on that dataset's
target instance set, 2$\times$2 small multiples. Two real bugs caught and
fixed during iteration, both by looking at an actual rendered image rather
than trusting the code: (1) marker size initially scaled with $n$, and
since $n$ also set the *legend* glyph size, Breast Cancer's $n{=}147$
methods produced $\sim$900pt legend markers that buried the real data under
the legend itself — fixed with a fixed marker size, since $n$ is redundant
with completeness $\times$ panel total anyway (and was later dropped from
the legend text entirely, same reasoning). (2) Methods with zero successes
on a panel's target set were initially omitted outright (no $\ell_0$ to
plot) — changed to an explicit hollow marker at $x{\approx}0$ (jittered per
zero-completeness method within a panel, so e.g. Australian's two zero-complete
NICE variants don't render exactly on top of each other) rather than
silently vanishing, since silent omission read as an oversight rather than
the real finding it is.

**Three genuinely zero-completeness cases**, each with a distinct root
cause (not "NICE/DiCE just didn't try hard enough" — a specific, different
failure status dominates each):

| dataset | method | dominant status | n |
|---|---|---|---|
| Australian | NICE(base) | `not_flipped` | 36/36 |
| Australian | NICE(sparse) | `timeout` | 36/36 |
| ILPD | DiCE | `dice_invalid_config` | 45/45 |

**Correction, caught by the user comparing the figure against the
recovery-rate table directly.** The plot's denominator was originally
PACE-relaxed's *own* `C_more_feats`-recovered set
(`pace_recovered_set`/`target_pairs_set`, e.g. 24 for ILPD, 51 for HELOC) —
the same quantity used for the L1/L2 baseline-comparison numbers above, by
design, for consistency with that table. But it's the wrong choice for
*this* figure specifically: it made PACE 100% complete on every panel by
tautology (the set is defined as "where PACE succeeded"), and silently
restricted every baseline's comparison to only the instances PACE itself
already solved — hiding how baselines do on the harder residual instances
PACE-relaxed doesn't recover either. For Breast Cancer and Australian this
was invisible (100% recovery there, so the two denominators coincide by
accident); for ILPD (53% recovery) and HELOC (91%) it was not, which is
what surfaced it.

**Fixed**: the denominator is now `target_failed_set` — *all* of
PACE-default's original failures for that dataset (45 for ILPD, 56 for
HELOC, matching the recovery-rate table exactly), computed directly from
`output_realdata.joblib`'s `pace` rows with `status != "ok"`, independent of
`l2_sweep_*.joblib`. PACE-relaxed's own completeness is now
`len(pace_recovered_set) / len(target_failed_set)` — genuinely informative
rather than tautological: **53.3% for ILPD, 91.1% for HELOC** (100% still
for Breast Cancer/Australian, correctly, since those really do recover
fully). Baselines are now scored against the full set too, so their raw
success counts changed (e.g. HELOC: DiCE 47$\to$52, NICE(base) 25$\to$29,
NICE(sparse) 39$\to$43, MCCE 39$\to$42 — all higher, since they get credit
for succeeding on the $\sim$5 extra instances PACE-relaxed itself doesn't
recover, which the old denominator excluded from consideration entirely).
Verified the corrected numbers reconcile exactly with each dataset's own
`RECOVERY RATE` table from the original SLURM job logs before re-sending
the figure. `l2_completeness_vs_sparsity.pdf` regenerated; the L1/L2
baseline-$\ell_0$-comparison numbers earlier in this section (the ones
quoted in the summary table and prose above) were **not** affected by this
bug — they already used the dataset-appropriate set consistently for both
numerator and denominator within each comparison, this was specifically a
bug in the later figure-building script's choice of denominator, introduced
when it was written, not a re-derivation of the earlier numbers.

### Why NICE(base) shows `not_flipped` on Australian: a NICE-specific float32 precision bug

User asked directly why NICE(base) — which is supposed to always return a
"justified" counterfactual (`justified_cf=True`, NICE's default) — could
ever fail to flip. Traced through both this project's wrapper
(`_e5_lib.py`) and the installed `nice` package's own source
(`nice/utils/data.py`) rather than reasoning from the algorithm's
description alone.

**The mechanism, precisely.** `nice.utils.data.data_NICE.__init__` computes
`train_proba = predict_fn(X_train)` **once**, as a single batched call over
the whole training set, at explainer-build time, and filters candidate
neighbours to `y_train == argmax(train_proba)` (correctly-classified
points) — this is what `justified_cf` actually checks. `nice_cf_one`
(`_e5_lib.py:929`) then **independently re-verifies** the returned
candidate for this benchmark's own status/`p_cf` bookkeeping:

```python
x_f_sc = np.asarray(x_f_sc, dtype=np.float32).reshape(-1)      # line 949
...
x_cf = np.asarray(x_cf, dtype=np.float32).reshape(-1)          # line 1064 (OHE-space path)
...
p_cf = float(predict_proba_target(x_cf[None, :])[0])
yhat_cf = int(p_cf >= proba_threshold)
if yhat_cf != y_desired:
    status = "not_flipped"
```

So the candidate is downcast to float32 **between** NICE's own internal
build-time check and this wrapper's final re-check — a step that doesn't
exist in NICE's own logic at all.

**Live confirmation** (`check_tabpfn_batch_context.py`, SLURM job 319461;
reproduced `australian`/TabPFN/seed=1/idx=2, one of the 36
Australian/nice\_base `not_flipped` instances on PACE's recovered set):

| evaluation | $p(\text{class}=1)$ | predicted class | agrees with target? |
|---|---|---|---|
| NICE's cached batched prediction (build time) | 0.662109 | 1 | yes |
| Fresh batched re-score (sanity check) | 0.662109 | 1 | yes |
| Solo re-score, full float64 | 0.662271 | 1 | yes |
| **Solo re-score, float32-cast** | **0.224216** | **0** | **no** |

Two effects are visible and separable: a genuine but tiny TabPFN
batch-context effect (0.662109 batched vs.\ 0.662271 solo alone — TabPFN's
in-context-learning predictions do shift slightly with batch composition,
confirmed directly, but far too small to flip anything on its own here),
and the float32 cast, which alone swings the probability by **0.44** and
flips the decision. A float64$\to$float32$\to$float64 round-trip should
only introduce $\sim$1e-7 relative error; a 0.44 absolute swing from that is
not explainable by rounding alone — it indicates TabPFN's learned decision
function is extremely locally sensitive at this specific point, i.e.\ this
lands in a genuinely sharp/unstable region of its decision surface, not
merely "a rounding bug happened to flip a coin-flip case."

**Checked whether this is systemic or NICE-specific** — grepped every
method's own code path for `float32`:

| method | separate re-verification after generation? | float32 cast before that re-check? |
|---|---|---|
| NICE (`nice_cf_one`) | yes | **yes — the bug** |
| PACE (`stageC_select_best`) | no (`status`/`p_cf` computed once, from the same batched float64 call that scores candidates; float32 cast at line 847 happens *after* the decision is locked in, on the output vector only) | n/a |
| DiCE (`dice_to_standard_out` $\to$ `validate_cf`) | yes | no — float64 preserved throughout both stages |
| MCCE (`mcce_cf.py`) | — | no `float32` anywhere in the file |

**Conclusion: this is specific to NICE's wrapper, not a systemic issue
quietly inflating failure rates elsewhere in the benchmark.** PACE and DiCE
structurally avoid it (PACE by never re-verifying post-cast; DiCE by never
casting down at all); MCCE doesn't cast at all. Not fixed in this pass — it
would change historical `not_flipped` counts for NICE(base)/NICE(sparse)
specifically and needs a decision on scope (real data only? mock too? worth
a full rerun given the magnitude found here?) before touching it.

**Text status:** not yet pasted into `pace.tex`. L1/L2's corrected framing
("relaxation recovers completeness; does not reliably preserve a sparsity
advantage") should replace the original spec's implicit "still sparser"
framing in Section 5; the completeness-vs-sparsity figure and the NICE
float32 finding are not yet placed in the manuscript.
**Not yet done:** decide on and implement a fix for the NICE float32 bug
(would require rerunning NICE-base/NICE-sparse, a non-trivial scope
decision); write the L1/L2 Section 5 text; place the new figure and its
caption.
