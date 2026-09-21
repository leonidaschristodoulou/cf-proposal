# Two reproducible TabPFN instability gotchas

**TL;DR.** TabPFN's predictions for a given row are not guaranteed to be
stable under two operations that are harmless for classical tabular models
(logistic regression, random forests, gradient-boosted trees): (1) casting
inputs from `float64` to `float32` and back, and (2) changing which other
rows are present in the same inference call. Both are demonstrated below
with a real, reproducible example where a rounding-level input perturbation
flips a prediction with a probability swing of 0.44 — far larger than the
perturbation itself should be able to cause under ordinary floating-point
error propagation.

Both issues share a root cause: TabPFN is an in-context-learning
transformer, not a row-independent function. Its prediction for one query
point is computed jointly with (and attends over) the training set and,
depending on how you call it, potentially other query points too — so
"the model's prediction for row $x$" is not as well-defined a quantity as it
is for classical models unless you're careful about exactly how $x$ is
presented to it.

---

## Issue 1: `float32` downcasting can flip predictions near the decision boundary

### The setup

Any pipeline that stores or serializes model inputs will, at some point,
tend to downcast to `float32` — for memory, for compatibility with some
downstream library, or just because `np.float32` is a common default. For
classical models this is inert: rounding a `float64` input to `float32`
precision changes each feature value by a relative error on the order of
$10^{-7}$, which propagates to a change in a linear model's or tree
ensemble's output on a similar order — utterly negligible for any decision
that isn't already balanced on a knife's edge to 7 decimal places.

### What actually happens with TabPFN

We reproduced this directly: fit `TabPFNClassifier` on a real, public
tabular dataset (Statlog/UCI "Australian Credit Approval," 690 rows, 14
numeric features, binary classification), identify a query point $x$ whose
predicted class matches a target label, and re-score that *exact same
point* after a `float64 → float32 → float64` round trip (i.e. the values are
only ever off by `float32` rounding error, never actually changed):

| evaluation | $p(\text{class}=1)$ | predicted class |
|---|---|---|
| Full `float64` | 0.662271 | 1 |
| After `float64 → float32 → float64` round trip | 0.224216 | 0 |

The predicted probability moves by **0.44**, and the decision flips. This
is not explainable by ordinary rounding error propagation — it indicates
that TabPFN's learned decision function is *locally extremely sensitive* at
this point, i.e. this specific point sits in a sharp, unstable region of the
model's decision surface, and an input perturbation small enough to be
invisible for a classical model is large enough to cross it.

### Why this happens

TabPFN is a large transformer that computes its prediction via attention
over the full training context, not a smooth closed-form function of the
input coordinates the way a linear model is. There is no guarantee that
small input perturbations produce proportionally small output changes
everywhere in input space — and near a decision boundary specifically,
where the model's output is (by definition) balanced between two classes,
the local behaviour of a high-capacity nonlinear function is exactly where
such sharp, non-smooth regions are most likely to be encountered.

### Practical implications

- **Don't assume dtype casts are free.** Any pipeline that scores a point
  once (in one precision) to select or generate it, and later re-verifies
  that same point in a different precision, should not assume the two
  checks will agree — especially for points close to the decision
  threshold, which is exactly the population you're most likely to be
  working with if you care about boundary-adjacent behaviour at all
  (counterfactual search, active learning near the margin, adversarial
  robustness checks, calibration analysis, etc.).
- **Keep a single consistent precision through the whole decision path.**
  If a value needs to be logged, stored, or serialized as `float32` for
  space reasons, make the actual accept/reject decision using the
  `float64` value that was *actually scored*, not by re-deriving it from
  the downcast copy.
- **This is a detection tool, not just a caveat.** Round-tripping a batch
  of borderline points through `float32` and checking which predictions
  flip is a cheap, direct way to find which of your points TabPFN
  considers "genuinely uncertain" versus "confidently placed but happens
  to be near 0.5" — the ones that flip under an imperceptible perturbation
  are the former.

---

## Issue 2: predictions depend on which other rows are in the same batch (in-context learning)

### The setup

For a row-independent classical model, `model.predict_proba(X)[i]` for a
given row `i` is the same regardless of what else is in `X` — scoring one
row alone or as part of a 10,000-row batch gives identical output for that
row (up to floating-point summation order, which is negligible). This
independence is usually assumed implicitly by anyone building a pipeline
around a scikit-learn-style classifier: fit once, predict on whatever
subset, in whatever order, whenever it's convenient.

### What actually happens with TabPFN

TabPFN's forward pass is a transformer that attends across the entire
context it's given — which, depending on how you call it, can include not
just the training set but the composition of the query batch itself. We
confirmed this directly on the same real point as above: its prediction
differs slightly depending on whether it's scored as part of a large batch
versus alone:

| evaluation | $p(\text{class}=1)$ |
|---|---|
| Scored as part of a full training-set batch | 0.662109 |
| Scored alone (single-row batch), same precision | 0.662271 |

The shift here (0.0002) is small — far too small to flip this particular
decision on its own — but it is real, reproducible, and entirely absent for
any classical tabular model. For a point sitting closer to a decision
threshold, or under a batch composition that differs more substantially
from the reference batch, there is no reason to expect this effect stays
small.

### Why this happens

This is a direct consequence of TabPFN's architecture, not a bug: in-context
learning means the model's representation of "what a class boundary looks
like" is constructed fresh, per forward pass, from whatever context
(training set, and depending on the calling convention, query set) is
presented alongside the point being scored. Change the context, and you are
in a meaningful sense asking a *different*, freshly-instantiated function to
make the prediction — even though the row you care about hasn't changed at
all.

### Practical implications

- **A cached "was this correctly classified" check can go stale.** If you
  compute predictions for your training/reference set once (e.g. to
  identify which points the model gets right, or to build a lookup/filter
  based on model behaviour) and later re-score a subset of those same
  points in a different batch — even with byte-identical feature values —
  don't assume the two will agree.
- **Batch composition is a hidden hyperparameter.** For any TabPFN-based
  system where an individual row's classification matters (not just
  aggregate accuracy), the batch it's evaluated in should be treated as
  part of the specification, not an implementation detail free to vary
  between calls.
- **This compounds with Issue 1.** In the reproduced example, the
  batch-context effect alone (0.662109 → 0.662271) doesn't change the
  decision; the float32 cast on top of it does (→ 0.224216). Two
  independently small, individually-inert-looking numerical choices
  combined to flip a real prediction. Neither alone would have been
  caught by a superficial check ("the batch is fine, it's a public API
  parameter"; "it's just a dtype cast, that's always safe").

---

## Issue 3: this shows up at scale in the actual PACE benchmark — apparent "zero-change" counterfactuals

### The setup

While rebuilding the ℓ0-vs-ℓ2 scatter plot for the paper's real-world
results (`output_realdata.joblib`), we found rows with `status == "ok"`,
`flip_ok == 1`, and `l0 == 0` — i.e. the returned counterfactual `x_cf` is
numerically identical to the factual `x_f` (zero features changed), yet the
row is marked as a successful, boundary-crossing counterfactual. This is
not a labeling bug: `p_f` and `p_cf` genuinely sit on opposite sides of 0.5
for the exact same input vector.

### What actually happens with TabPFN

Confirmed directly from the data: 434 such rows, **all on TabPFN, on no
other classifier**, across three of the five CE methods:

| method | count |
|---|---:|
| pace | 394 |
| nice_spars | 39 |
| nice_base | 1 |

Example rows (`x_f == x_cf` exactly, `flip_ok == 1`):

| method | $p_f$ | $p_{cf}$ | margin |
|---|---:|---:|---:|
| pace | 0.447754 | 0.566406 | 0.066406 |
| pace | 0.462271 | 0.507812 | 0.007812 |
| pace | 0.579937 | 0.485840 | 0.014160 |

### Why this happens (architecturally consistent with Issue 2)

For PACE specifically, this is explained directly by the benchmark's own
call pattern in `run_benchmark` (`_e5_lib.py`): `p_f` is computed with a
**solo** call, `p_f = float(pp(x_f[None, :])[0])`, while `p_cf` for the
returned candidate comes out of Stage C's **single batched call over the
full candidate set** (the architectural feature the paper's own efficiency
claim, M6, rests on). So the literally-identical input vector is scored
twice under two different batch contexts — exactly Issue 2's mechanism,
just now observed as a real consequence in the production benchmark rather
than a constructed demo. For `nice_spars`/`nice_base` the same symptom
appears, but we have not independently traced their call sites to confirm
the same batch-context cause applies there rather than, e.g., a float32
round-trip (Issue 1) somewhere in their own candidate-scoring path — the
symptom is confirmed, the exact mechanism per method is not.

### Practical implications

- **A single-instance rerun for illustration purposes should expect this.**
  When manually reproducing one factual's result (e.g. for a qualitative
  example figure), a fresh call may not reproduce the master joblib's exact
  `l0`/`p_cf`, and a `flip_ok=1, l0=0` case in particular should not be
  read as "PACE proposed doing nothing" — it's a batch-context artifact of
  how the target model was called, not a proposal PACE actually made.
- **This is now handled explicitly, not silently, in the paper's own
  analysis notebook** (`cfprop_analysis.ipynb`, ℓ0-vs-ℓ2 scatter section):
  `l0 == 0` rows are filtered out before plotting, with the reason stated
  inline, rather than left in as if they were genuine zero-change
  recourses.
- **Worth checking whether this affects reported completeness numbers.**
  These rows are counted as `flip_ok=1`/`status="ok"` successes in the
  completeness/reliability tables (Section 14, E2) — which is defensible
  (the row genuinely met the recorded success criterion at the time it was
  scored) but is a real, quantifiable source of PACE's TabPFN completeness
  being inflated by up to 394/4870 ≈ 8.1 percentage points from this effect
  alone, if one takes the position that a zero-change "success" shouldn't
  count. Not resolved here — flagged for whoever decides how strict the
  completeness definition should be.

---

## Reproducing this

The check is cheap and architecture-agnostic — no special setup beyond a
fitted `TabPFNClassifier` and a point you want to stress-test:

```python
import numpy as np
from tabpfn import TabPFNClassifier

# model = TabPFNClassifier(device="auto").fit(X_train, y_train)
# x = a query point of interest (float64 ndarray, shape (n_features,))

# --- Issue 1: float32 round-trip ---
p_f64 = model.predict_proba(x.astype(np.float64)[None, :])[0, 1]
p_f32 = model.predict_proba(x.astype(np.float32).astype(np.float64)[None, :])[0, 1]
print(f"float64: {p_f64:.6f}   float64-via-float32: {p_f32:.6f}   "
      f"flipped: {(p_f64 >= 0.5) != (p_f32 >= 0.5)}")

# --- Issue 2: batch context ---
p_batched = model.predict_proba(X_train)[row_index_of_x, 1]   # x scored inside a big batch
p_solo    = model.predict_proba(x[None, :])[0, 1]             # x scored alone
print(f"batched: {p_batched:.6f}   solo: {p_solo:.6f}   "
      f"flipped: {(p_batched >= 0.5) != (p_solo >= 0.5)}")
```

Running this over a population of points near your decision threshold (not
just one) will tell you how common these flips are for your specific model
and dataset — in the case that motivated this note, both were rare overall
but concentrated almost entirely on the hardest, most boundary-adjacent
instances, which is exactly where a practitioner is least likely to be
using classical intuitions about numerical stability.
