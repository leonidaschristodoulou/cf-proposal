# M4 — Define every symbol currently used undefined

**Grounding note:** all formulas below are read directly from `pfn_cf_guided_onehot.py`
(`stageA_build`, `stageB_generate`, `_perm_importance_proba_drop`). One value
(`boundary_k`/$k_B$) I could only confirm as `stageA_build`'s own default (200) — the
notebook's `guided_cf_build_cache` (which actually calls `stageA_build` at benchmark
time) lives in `tabpfn_cf5.ipynb` cell 9, not in this file, and I haven't confirmed it
doesn't override `boundary_k`. Flagged below — please cross-check against Table 1.
Assumes M1's definitions (feature units $U$, $\tau$, candidate set $C$, $M$, $s$,
$q_{\mathrm{low}}/q_{\mathrm{high}}$) are already in place.

---

### Importance weights: $I(u)$ and $\gamma$

Stage A computes a raw importance score $I(u)$ for each feature unit $u \in U$ via
**permutation importance on the predicted probability**: permute unit $u$'s value(s)
across a reference sample (categorical units are permuted as a block, preserving
one-hot validity) and measure the mean absolute change in $f_{\mathrm{guide}}(x)$
(the guidance model's output — see M6), averaged over `perm_repeats`$= 8$ repeats.

The **importance-sharpening exponent** $\gamma$ then converts raw importance into
sampling weight:
$$
w(u) \;\propto\; \max(I(u), \varepsilon)^{\gamma}, \qquad \varepsilon = 10^{-12},
$$
normalized so $\sum_{u \in U} w(u) = 1$. **Value: $\gamma = 1.0$** (`stageA_build`'s
`gamma` parameter — confirmed this is left at its default in the benchmark, i.e. no
sharpening beyond linear scaling by raw importance). $\gamma > 1$ would concentrate
sampling mass on the highest-importance units; $\gamma < 1$ would flatten it toward
uniform. Table 1 currently has no entry for $\gamma$ — add $\gamma = 1.0$.

### Interpolation and "constrained interpolation"

Each candidate proposal first samples a small set of feature units to change (this
*is* the "constrained" part):
$$
k \sim \mathrm{Unif}\{1, \dots, s\}, \quad\text{with probability } p_{ch}\text{; else } k = 0,
$$
then draws $k$ **distinct units without replacement**, weighted by $w(u)$ above. So
"constrained interpolation" means: interpolation is applied only to this
randomly-sampled, importance-weighted subset of at most $s$ units — never to the full
feature vector — and with probability $1 - p_{ch}$, no interpolation happens at all for
that candidate (yielding $x_f$ itself as the proposal, before Stage C's flip check
filters it out). **Value: $p_{ch} = 0.8$** (`p_change`).

For the chosen units, a single **interpolation weight** $\alpha$ is drawn once per
candidate,
$$
\alpha \sim \mathrm{Unif}(\alpha_{\min}, \alpha_{\max}), \qquad [\alpha_{\min},
\alpha_{\max}] = [0.5,\, 1.0] \ \text{(benchmark value, matches Table 1's Unif}[0.5,1.0]\text{)},
$$
and applied differently by unit type:
- **numeric unit** $u = \{j\}$: $x_j \leftarrow (1-\alpha)\,x_{f,j} + \alpha\,d_j + \eta$,
  a convex combination of the factual's value and the donor $d$'s value (donor = the
  anchor or boundary point for that proposal), plus Gaussian jitter $\eta \sim
  \mathcal{N}(0, \sigma)$ (see next section for $\sigma$).
- **categorical unit** (one-hot group): the donor's one-hot block is copied wholesale,
  $x_{\mathrm{cols}(u)} \leftarrow d_{\mathrm{cols}(u)}$ — interpolation isn't
  meaningful between one-hot levels, so the operator adopts the donor's category
  outright rather than blending.

This directly answers R1's "how is constrained interpolation implemented": it is a
per-candidate, importance-weighted random subset of $\le s$ units, each moved a
shared-$\alpha$ fraction of the way from $x_f$ toward a donor (numeric) or set equal to
the donor (categorical) — not a global interpolation over all features.

### Noise scale: $\sigma_{\mathrm{base}}$ (and reconciling "$c$")

The guided-noise operator's Gaussian scale is instance-adaptive:
$$
\sigma = \mathrm{clip}\big(\sigma_{\mathrm{base}} \cdot \mathrm{scale},\ \sigma_{\min},\ \sigma_{\max}\big),
\qquad \mathrm{scale} = \mathrm{clip}\!\left(\frac{|f(x_f) - \tau|}{0.25},\ 0.2,\ 2.5\right),
$$
i.e. factuals closer to the decision boundary get proportionally smaller noise, and
factuals far from it get larger noise, within $[\sigma_{\min}, \sigma_{\max}]$.
**Values: $\sigma_{\mathrm{base}} = 0.25$, $\sigma_{\min} = 0.05$, $\sigma_{\max} =
1.0$** (`base_sigma`/`sigma_min`/`sigma_max`). **Table 1's $\sigma_{\mathrm{base}} =
0.25$ and Algorithm 1's "$c$" are the same quantity** — there is no separate `c` in the
code. State explicitly $c := \sigma_{\mathrm{base}}$, or simply rename Algorithm 1's
$c$ to $\sigma_{\mathrm{base}}$ throughout.

### Pool sizes: $k_A$, $k_B$

- $k_A$ (anchor pool size): number of training points, already on the desired side of
  the boundary, eligible as anchor donors. **Benchmark value: $k_A = 250$**
  (`guided_anchor_k` as passed to `guided_cf_build_cache` in `run_benchmark`).
- $k_B$ (boundary pool size): number of training points nearest the decision boundary
  ($f(\cdot) \approx \tau$), eligible as boundary donors, balanced between the two
  predicted classes when possible. **Value: $k_B = 200$** — this is `stageA_build`'s
  own default; *please verify the notebook's `guided_cf_build_cache` (cell 9) doesn't
  override it*, since I could only confirm the default in `pfn_cf_guided_onehot.py`
  itself, not the call site.

---

**Summary table of values to reconcile with your existing Table 1:**

| symbol | meaning | value | source |
|---|---|---|---|
| $\pi_a, \pi_b, \pi_n$ | anchor/boundary/noise mixture weights | $0.50, 0.35, 0.15$ | `stageB_generate` call args (M1) |
| $M$ | candidate set size | $1000$ | `run_benchmark`'s `guided_n_candidates` (function default: 800) |
| $s$ | max changed feature units | $8$ | `run_benchmark`'s `guided_max_changed_features` (function default: 3) |
| $\gamma$ | importance-sharpening exponent | $1.0$ | `stageA_build`'s `gamma` |
| $p_{ch}$ | probability a candidate changes anything | $0.8$ | `p_change` |
| $[\alpha_{\min}, \alpha_{\max}]$ | interpolation weight range | $[0.5, 1.0]$ | `alpha_range` |
| $\sigma_{\mathrm{base}}$ (a.k.a. "$c$") | base noise scale | $0.25$ | `base_sigma` |
| $\sigma_{\min}, \sigma_{\max}$ | noise scale clip bounds | $0.05, 1.0$ | `sigma_min`/`sigma_max` |
| $k_A$ | anchor pool size | $250$ | `guided_anchor_k` |
| $k_B$ | boundary pool size | $200$ (unverified at call site) | `stageA_build`'s `boundary_k` default |
| $q_{\mathrm{low}}, q_{\mathrm{high}}$ | quantile-clip bounds | $0.01, 0.99$ | `q_low`/`q_high` |
