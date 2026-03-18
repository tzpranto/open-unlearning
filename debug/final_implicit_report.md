# Final Implicit-Correction Report (SIBL Unlearning)

## 1) Why implicit correction is needed in this unlearning setup

In our bilevel unlearning setup, we optimize two competing goals:

- **Outer objective (forget):** push the model to remove target information.
- **Inner objective (retain):** keep utility on retained data within a constraint budget.

Without implicit correction, the outer update treats the inner problem as if it were static.  
But in bilevel optimization, changing parameters for forgetting also changes the inner optimum.  
So the true "best" outer direction should account for how the retain-optimal point moves.

Intuition:

- A naive outer gradient says "push hard to forget."
- Implicit correction says "push to forget, but discount directions that will badly damage the retain-constrained inner solution."

This is exactly why implicit correction matters for stable unlearning: it is a second-order sensitivity correction to avoid over-aggressive forgetting steps that immediately break utility constraints.

---

## 2) How the CG-based implicit solver was implemented (high-level)

The CG implementation follows the standard implicit-diff recipe:

1. Build the right-hand side vector \(v\) from the masked outer gradient.
2. Define a Hessian-vector product operator for the inner objective (with damping).
3. Solve approximately for \(h\) in:
   \[
   (H_{\text{inner}} + \lambda I) h \approx v
   \]
   using Conjugate Gradient.
4. Compute a correction term via an outer-Hessian-vector product.
5. Apply corrected gradient update in the outer step.

Everything is done with HVPs (no explicit Hessian matrix), which is computationally feasible for large models.

---

## 3) Why CG did not work well in practice

Even though CG is principled, we observed poor behavior in our setting:

- High linear-system residuals in several blocks (sometimes very large).
- Strong sensitivity to damping and local scale of the linear solve.
- Constraint residual \(r\) and dual variable \(\lambda\) could grow quickly in unstable runs.
- In blockwise experiments, CG could produce very poor solve quality on some blocks even when others looked acceptable.

Empirically, this translated to unstable retain behavior and weaker practical unlearning trade-offs versus the best Neumann runs.

---

## 4) Ill-conditioning check: Rayleigh proxy and what we found

To check whether the system is fundamentally ill-conditioned, we used a lightweight Rayleigh proxy:

- **Rayleigh value (definition):** for a direction \(q\),
  \[
  R(q) = \frac{q^\top H q}{q^\top q}
  \]
  and when \(q\) is normalized, \(R(q) = q^\top H q\).
  Intuitively, this is the "effective strength" of \(H\) along direction \(q\).

- Sample random normalized probe vectors \(q\).
- Compute \(q^\top H q\) (via HVP).
- Track min/max Rayleigh values, ratio, and number of non-positive probes.

Interpretation:

- Very large ratio or many non-positive probes would suggest problematic local geometry in the solve.
- Moderate ratio and mostly positive probes suggest conditioning may not be the main bottleneck.

Findings:

- We did **not** get strong evidence that severe ill-conditioning alone explains failures.
- The dominant issue looked more like **solver-quality instability** (method/config mismatch), especially under aggressive update settings.

---

## 5) How Neumann implicit correction was implemented

We implemented a truncated Neumann-style inverse approximation with damping:

- Define a damped masked operator for the inner linear-response map.
- Approximate inverse action for \(H^{-1}v\) iteratively.
- Build corrected outer gradient from \(v\) and an outer-HVP correction term.
- Add safeguards:
  - non-finite checks,
  - correction-growth fallback,
  - adaptive alpha backtracking when corrections explode.

This made the solver much more robust than naive first versions.

### Parameters used in our Richardson-Neumann block

- \(v\): masked outer gradient (right-hand side).
- \(H_{\text{in}}\): inner-objective Hessian-vector product operator (implemented via autograd HVP).
- \(\mu\): damping coefficient in \(H_{\text{in}} + \mu I\).
- \(\alpha\): update step size for Richardson fixed-point iteration.
- \(J\): number of Neumann/Richardson steps.
- `max_growth_ratio`: reject correction if \(\|g_{\text{corr}}\|\) is too large relative to \(\|v\|\).
- `backtrack_factor`, `backtrack_max_tries`: reduce \(\alpha\) when the correction is unstable.

### Richardson method (pseudocode we are currently using)

```text
Input: v, H_in(.), H_alm(.), mask, mu, alpha0, J
for attempt = 0 .. backtrack_max_tries-1:
    alpha = clip(alpha0 * backtrack_factor^attempt, alpha_min, alpha_max)
    h = 0
    for t = 0 .. J:
        # Damped masked operator
        H_tilde(h) = mask * H_in(mask * h) + mu * (mask * h)
        residual = v - H_tilde(h)
        if residual or h is non-finite: mark unstable and break
        h = h + alpha * residual

    if unstable: continue
    c = mask * H_alm(mask * h)
    g_corr = v - c
    if g_corr non-finite: continue
    if ||g_corr|| > max_growth_ratio * ||v||: continue
    accept g_corr and stop

if no attempt accepted:
    fallback to g_corr = v
```

### Pseudocode in plain language

- We start from the current outer-gradient signal \(v\).
- We try to compute a corrected direction by repeatedly refining a helper vector \(h\).
- If the correction becomes numerically unsafe (too large or non-finite), we do not trust it.
- Instead, we reduce step size \(\alpha\) and try again.
- If all retries fail, we safely fall back to the original \(v\) update.

This keeps training moving while avoiding catastrophic correction steps.

### Backtracking: what, why, and how

- **What is backtracking?**  
  A safety mechanism that shrinks step size \(\alpha\) when an attempted correction is unstable.

- **Why do we need it?**  
  A step size that is fine on one block/iteration can be too aggressive on another.  
  Without backtracking, one bad step can explode the correction and destabilize training.

- **How are we doing it?**  
  We start from a base \(\alpha\), then retry with smaller values:
  \[
  \alpha,\ \alpha \cdot \beta,\ \alpha \cdot \beta^2,\dots
  \]
  where \(\beta \in (0,1)\) is `backtrack_factor` and number of retries is `backtrack_max_tries`.  
  We accept the first stable correction; otherwise we fall back to \(v\).

---

## 6) Why Neumann failed first (whole-model version)

The first major failure mode was not "math wrong," but **memory/computation shape**:

- Whole-model implicit path needed huge flattened vectors.
- This triggered GPU OOM during vector materialization in realistic runs.

So even before convergence quality, full implicit correction was operationally infeasible at that scale.

---

## 7) How blockwise implicit was designed

To remove the full-vector bottleneck, we switched to blockwise implicit correction:

- Partition trainable parameters into layer-based blocks (e.g., last layers).
- Solve implicit correction per block.
- Write corrected gradients back block-by-block.

Benefits:

- Avoids full flatten OOM.
- Gives per-block diagnostics (residuals, correction norms, fallback events).
- Lets us localize unstable regions instead of treating the whole model as one giant system.

---

## 8) Why we switched to Richardson-style Neumann

We initially had a legacy Neumann accumulation that was less faithful to solving \(Hh=v\).
We then switched to a Richardson fixed-point form, which is a more direct iterative solve for inverse action.

Why this change:

- In offline random linear-system tests, legacy Neumann had consistently worse residuals.
- Richardson-style updates were better behaved and more interpretable as a proper fixed-point solve.
- This aligned better with our practical objective: stable approximate implicit correction.

---

## 9) Current status: promising but not fully solved

Current results suggest Neumann is promising:

- Preliminary numbers show SIBL+Neumann outperforming SIBL+CG in this setup.
- Run stability improved with blockwise mode, conservative damping, and alpha backtracking.

But open issues remain:

- Some blocks still trigger exploding-correction fallbacks.
- Alpha often needs backtracking (base step is still aggressive in certain states).
- Retain-loss and dual-variable drift can still appear in longer runs.
- Stability is heterogeneous across blocks and outer iterations.

So the direction is good, but we still need stronger robustness in hard regions.

---

## 10) Next plan: surgical bilevel updates via mechanistic core regions

Our next step is to move from broad sparsity to **surgical targeting**:

1. Use off-the-shelf mechanistic interpretability tools to identify core influential regions/layers for:
   - forget behavior,
   - retain behavior.
2. Allow overlap between the two sets.
3. Set forgetting aggressiveness based on overlap and influence confidence.
4. Run bilevel implicit correction only on selected high-impact regions (instead of broad/global masks).

Expected upside:

- Better forget/retain trade-off,
- less unnecessary parameter movement,
- improved solver stability by focusing on lower-dimensional, semantically relevant regions.

---

## 11) Brief quantitative snapshot (current)

- **SIBL + Neumann (promising):** stronger utility/forget trade-off than current CG baseline in your latest runs.
- **SIBL + CG:** currently underperforming with weaker stability and poorer practical results.

This supports keeping Neumann as the primary implicit direction while we improve robustness and move to surgical targeting.

---

## 12) Off-the-shelf bilevel tools (current note)

I am exploring these off-the-shelf bilevel tools and how to integrate them into our system:

- **JAXopt**
- **TorchOpt**
- **higher**

References (verified):

- TorchOpt documentation: https://torchopt.readthedocs.io/en/latest/
- higher repository: https://github.com/facebookresearch/higher
- JAXopt implicit differentiation docs: https://jaxopt.github.io/stable/implicit_diff.html
- Lorraine et al., *Optimizing Millions of Hyperparameters by Implicit Differentiation* (AISTATS 2020): https://proceedings.mlr.press/v108/lorraine20a.html
- Blondel et al., *Efficient and Modular Implicit Differentiation* (arXiv): https://arxiv.org/abs/2105.15183

