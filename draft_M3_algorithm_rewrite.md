## Algorithm 3 (Stage B: Candidate generation) — corrected

Two fixes folded in: (1) immutable units are excluded from the eligible set *before*
any sampling happens (structural, not a downstream filter — matches `stageB_generate`'s
`units` construction exactly: immutable units are simply never appended to the list
`sample_units` draws from); (2) $\sigma$ is now computed as an explicit step from
$f_{\mathrm{guide}}$, $\tau$, and the three scale parameters, instead of arriving as a
pre-baked `\Require` input.

```latex
\begin{algorithm}[htbp]
\caption{Candidate generation}
\label{alg:stageB_candidates}
\begin{algorithmic}[1]
\Require $x_f$, guidance model $f_{\mathrm{guide}}$, weights $w(u)$ for $u \in U$,
boundary pool $B$, anchor pool $A$, immutable units $U_{\mathrm{imm}}$, candidate
budget $M$, mixture weights $\pi_a,\pi_b,\pi_n$, max changed units $s$, change
probability $p_{ch}$, interpolation range $[\alpha_{\min},\alpha_{\max}]$, threshold
$\tau$, noise-scale parameters $\sigma_{\mathrm{base}}, \sigma_{\min}, \sigma_{\max}$
\Ensure candidate set $C$ with $|C| = M$
\State $U_{\mathrm{act}} \gets U \setminus U_{\mathrm{imm}}$ \Comment{only mutable units are ever eligible for change}
\State $w(u) \gets w(u) \big/ \sum_{u' \in U_{\mathrm{act}}} w(u')$ for $u \in U_{\mathrm{act}}$ \Comment{renormalize over $U_{\mathrm{act}}$; every \textsc{sample\_units} call below draws only from $U_{\mathrm{act}}$}
\State $\mathrm{margin} \gets |f_{\mathrm{guide}}(x_f) - \tau|$
\State $\mathrm{scale} \gets \mathrm{clip}(\mathrm{margin} / 0.25,\ 0.2,\ 2.5)$
\State $\sigma \gets \mathrm{clip}(\sigma_{\mathrm{base}} \cdot \mathrm{scale},\ \sigma_{\min},\ \sigma_{\max})$ \Comment{instance-adaptive noise scale, shared by all three proposal types below}
\State $M_a \gets \mathrm{round}(M \cdot \pi_a)$, $M_b \gets \mathrm{round}(M \cdot \pi_b)$, $M_n \gets M - M_a - M_b$
\If{$A = \emptyset$} \State $M_b \gets M_b + M_a$; $M_a \gets 0$ \EndIf
\State $C \gets \emptyset$
\For{$i = 1$ to $M_a$} \Comment{anchor candidates}
  \State $\mathrm{donor} \gets \mathrm{uniform\_random\_choice}(A)$
  \State $\mathrm{chosen} \gets \mathrm{sample\_units}(w, s, p_{ch})$
  \State $x \gets \mathrm{constrained\_interpolate}(x_f, \mathrm{donor}, \mathrm{chosen}, [\alpha_{\min},\alpha_{\max}], \sigma)$
  \State $C \gets C \cup \{\, \mathrm{clip\_to\_quantiles}(x, \mathrm{chosen}) \,\}$\label{ln:clip-anchor}
\EndFor
\For{$i = 1$ to $M_b$} \Comment{boundary candidates}
  \State $\mathrm{donor} \gets \mathrm{uniform\_random\_choice}(B)$
  \State $\mathrm{chosen} \gets \mathrm{sample\_units}(w, s, p_{ch})$
  \State $x \gets \mathrm{constrained\_interpolate}(x_f, \mathrm{donor}, \mathrm{chosen}, [\alpha_{\min},\alpha_{\max}], \sigma)$
  \State $C \gets C \cup \{\, \mathrm{clip\_to\_quantiles}(x, \mathrm{chosen}) \,\}$\label{ln:clip-boundary}
\EndFor
\For{$i = 1$ to $M_n$} \Comment{guided-noise candidates}
  \State $\mathrm{chosen} \gets \mathrm{sample\_units}(w, s, p_{ch})$
  \State $x \gets \mathrm{guided\_noise}(x_f, \mathrm{chosen}, \sigma)$
  \State $C \gets C \cup \{\, \mathrm{clip\_to\_quantiles}(x, \mathrm{chosen}) \,\}$\label{ln:clip-noise}
\EndFor
\State \Return $C$ \Comment{$|C| = M$}
\end{algorithmic}
\end{algorithm}
```

**Two things to check against your actual notation before pasting in:**
1. If $U$ and $w(u)$ are passed into this algorithm already restricted to actionable
   units elsewhere in your pipeline (e.g. if Stage A itself excludes $U_{\mathrm{imm}}$
   when it builds $w$), lines 1–2 here would be redundant — worth a quick check of
   Stage A's own code/pseudocode for whether `unit_weights` is computed over *all* of
   $U$ or already only $U_{\mathrm{act}}$. From what I read in `stageA_build`, Stage A's
   `_perm_importance_proba_drop` computes importance over *all* units (it has no
   visibility into immutability), and the actionable-only subsetting happens inside
   `stageB_generate` itself — so keeping lines 1–2 here, rather than moving them to
   Stage A, matches the code as written.
2. `\State $\mathrm{margin} \gets ...$` reuses $f_{\mathrm{guide}}(x_f)$ as a fresh
   call — if your Stage A pseudocode already computed and cached
   $p = f_{\mathrm{guide}}(X_{\mathrm{train}})$, this is a *separate* singleton call on
   just $x_f$ (not a lookup into that cached array), since $x_f$ generally isn't a
   training point. Worth a one-word clarification in prose if a reader might conflate
   the two.
