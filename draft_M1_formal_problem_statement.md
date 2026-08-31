# M1 — Formal problem statement (drop-in before Algorithm 1)

**Grounding note:** every symbol below is checked against `pfn_cf_guided_onehot.py`
(`stageA_build`, `stageB_generate`, `stageC_select_best`). Where the function's own
default differs from what `run_benchmark` actually passes in (i.e. what produced your
reported results), I used the benchmark's value and flagged it — cross-check against
your existing Table 1, since I don't have the manuscript source to confirm it matches.

---

### Problem statement

Let $f : \mathbb{R}^d \to [0,1]$ be a binary probabilistic predictor operating on a
$d$-dimensional model-space representation (numeric features standardized, categorical
features one-hot encoded), with decision threshold $\tau$ (we use $\tau = 0.5$
throughout). Given a factual instance $x_f \in \mathbb{R}^d$ with predicted label
$\hat{y}_f = \mathbb{1}[f(x_f) \ge \tau]$ and a desired class $y_{\mathrm{des}} = 1 -
\hat{y}_f$, a **counterfactual explanation** is a point $x_{cf} \in \mathbb{R}^d$ such
that $\mathbb{1}[f(x_{cf}) \ge \tau] = y_{\mathrm{des}}$, subject to feasibility
constraints on which coordinates of $x_f$ may change and by how much.

**Feature units.** Coordinates of $x \in \mathbb{R}^d$ are grouped into *feature
units* $U = \{u_1, \dots, u_m\}$: each numeric feature is its own singleton unit, and
each one-hot-encoded categorical feature's dummy columns form a single unit (so that a
"one-unit change" to a categorical feature means switching its active level, not
independently perturbing individual dummy columns). $U$ partitions into a mutable set
$U_{\mathrm{mut}}$ and an immutable set $U_{\mathrm{imm}} = U \setminus U_{\mathrm{mut}}$,
supplied per-dataset (e.g. protected or structurally fixed attributes).

**Proposal space.** Given the factual $x_f$ and a reference pool $R \subseteq X_{\mathrm{train}}$
(the training set in model space), PACE constructs candidates via three *proposal
types*, each a map $(x_f, R) \mapsto x_{cand} \in \mathbb{R}^d$:

- **anchor**: $x_{cand} \in R$ is a training point already on the desired side of the
  boundary ($f(x_{cand})$ consistent with $y_{\mathrm{des}}$), used directly or as an
  interpolation endpoint.
- **boundary**: $x_{cand}$ is drawn from a pool of training points near the decision
  boundary ($f(\cdot) \approx \tau$), then interpolated toward $x_f$.
- **guided noise**: $x_{cand} = x_f$ perturbed along a subset of mutable units, with
  per-unit perturbation scale and unit-selection probability governed by Stage A's
  importance weights (below).

Each proposal type draws with probability $\pi_a, \pi_b, \pi_n$ respectively (mixture
weights over anchor/boundary/noise; **benchmark values** $\pi_a = 0.50$, $\pi_b =
0.35$, $\pi_n = 0.15$, i.e. `frac_anchor_mix`/`frac_boundary_mix`/`frac_guided_noise`
in `stageB_generate`), until a **candidate set** $C = \{x_{cand}^{(1)}, \dots,
x_{cand}^{(M)}\}$ of size $M$ is assembled (**benchmark value** $M = 1000$, i.e.
`guided_n_candidates` as called from `run_benchmark`; the function's own default is
800).

**Feasibility constraints.** A candidate $x_{cand} \in C$ is *feasible* if:
(i) it flips the predicted class, $\mathbb{1}[f(x_{cand}) \ge \tau] = y_{\mathrm{des}}$;
(ii) every categorical unit's one-hot block is valid (exactly one active level);
(iii) it leaves every $u \in U_{\mathrm{imm}}$ unchanged, and every changed numeric
unit stays within its training-quantile range (clipped to
$[q_{0.01}, q_{0.99}]$ per feature, `clip_to_train_quantiles`/`q_low`/`q_high`); and
(iv) it changes at most $s$ feature units from $x_f$ (**benchmark value** $s = 8$, i.e.
`guided_max_changed_features`; the function's own default is 3).

**Selection rule.** Among the feasible subset $C_{\mathrm{feas}} \subseteq C$, PACE
returns
$$
x_{cf} = \operatorname*{arg\,lex\,min}_{x \in C_{\mathrm{feas}}} \big(\, \ell_0(x, x_f),\ \ell_2(x, x_f),\ |f(x) - \tau| \,\big),
$$
a **lexicographic** minimization: first by $\ell_0$ (number of changed feature units,
group-aware for one-hot blocks), then by $\ell_2$ (Euclidean distance in model space)
to break ties, then by proximity of $f(x)$ to the threshold as a final tiebreaker. If
$C_{\mathrm{feas}} = \emptyset$, PACE returns $\varnothing$ (no counterfactual found).
(Verified directly in `stageC_select_best`: `order = np.lexsort((np.abs(pf - 0.5), l2, l0))`
— note `np.lexsort` sorts by its *last* key first, so this is exactly $\ell_0 \to \ell_2
\to |f(x)-\tau|$.)

Only after this point should Algorithm 1's pseudocode appear (M3).

---

**Symbols introduced here, for cross-reference with M4's fuller symbol pass:**
$x_f$, $f$, $\tau$, $y_{\mathrm{des}}$, $U$/$U_{\mathrm{mut}}$/$U_{\mathrm{imm}}$, $R$,
the three proposal types, $\pi_a/\pi_b/\pi_n$, $C$, $M$, feasibility (i)-(iv), $s$,
$q_{\mathrm{low}}/q_{\mathrm{high}}$, $\ell_0$, $\ell_2$, the lexicographic rule.
M4 still needs: $\alpha$/$[\alpha_{\min}, \alpha_{\max}]$ (interpolation weight for
anchor/boundary), $\gamma$ (importance-sharpening exponent), $c$ vs $\sigma_{\mathrm{base}}$
(noise scale — confirmed these are the *same* quantity, `base_sigma` in code; Algorithm
1's "$c$" should be renamed to $\sigma_{\mathrm{base}}$ or explicitly defined as
$c := \sigma_{\mathrm{base}}$), $k_A$/$k_B$ (anchor/boundary pool sizes), $p_{ch}$
(per-unit change probability in guided noise). I have code-verified values for all of
these ready whenever you want the M4 draft next — happy to do that now too.
