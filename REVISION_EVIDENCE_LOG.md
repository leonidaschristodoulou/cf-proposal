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
  Output: `dice_tabpfn_corrected_mock.joblib`.
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

## E2 — NICE-sparse `not_run` characterization

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

**Full picture across all 5 originally-skipped combos:**
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
4. **Merge corrected DiCE+TabPFN rows into the master joblib caches: COMPLETE.**
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
5. **E2's full status-normalization script** — not yet built; this log's NICE-sparse
   findings and the decided timeout value are direct inputs to it.
