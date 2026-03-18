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

This is exactly why implicit correction matters for stable unlearning: it is a curvature-aware correction to avoid over-aggressive forgetting steps that immediately break utility constraints.

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
- Strong sensitivity to damping and curvature scale.
- Constraint residual \(r\) and dual variable \(\lambda\) could grow quickly in unstable runs.
- In blockwise experiments, CG could produce very poor solve quality on some blocks even when others looked acceptable.

Empirically, this translated to unstable retain behavior and weaker practical unlearning trade-offs versus the best Neumann runs.

---

## 4) Ill-conditioning check: Rayleigh proxy and what we found

To check whether the system is fundamentally ill-conditioned, we used a lightweight Rayleigh proxy:

- Sample random normalized probe vectors \(q\).
- Compute \(q^\top H q\) (via HVP).
- Track min/max Rayleigh values, ratio, and number of non-positive probes.

Interpretation:

- Very large ratio or many non-positive probes would suggest problematic curvature.
- Moderate ratio and mostly positive probes suggest conditioning may not be the main bottleneck.

Findings:

- We did **not** get strong evidence that severe ill-conditioning alone explains failures.
- The dominant issue looked more like **solver-quality instability** (method/config mismatch), especially under aggressive update settings.

---

## 5) How Neumann implicit correction was implemented

We implemented a truncated Neumann-style inverse approximation with damping:

- Define a damped masked operator for inner curvature.
- Approximate inverse action for \(H^{-1}v\) iteratively.
- Build corrected outer gradient from \(v\) and an outer-HVP correction term.
- Add safeguards:
  - non-finite checks,
  - correction-growth fallback,
  - adaptive alpha backtracking when corrections explode.

This made the solver much more robust than naive first versions.

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

## 12) Off-the-shelf bilevel solvers: should we use them?

Short answer: **yes for prototyping and verification, not as a full drop-in replacement yet**.

Reason:

- Off-the-shelf packages are strong at the bilevel math plumbing (unrolling or implicit gradients),
- but our pipeline has custom pieces (masked/blockwise updates, ALM dual dynamics, memory-constrained HVP design, custom forget/retain objectives) that still require method-specific engineering.

Good candidates:

1. **TorchOpt** (PyTorch): explicit + implicit differentiation, including linear-system based implicit gradients.  
2. **higher** (PyTorch): differentiable unrolled optimizers (useful for "differentiate through K inner steps" baselines).  
3. **JAXopt** (JAX): mature implicit-diff API with custom root/fixed-point decorators and clear argmin-diff interfaces.

How these solvers compute gradient/update (intuition):

- **Unrolled differentiation:** run inner optimizer for K steps, then backprop through those K updates.
- **Implicit differentiation:** treat inner optimum as satisfying a stationarity equation, then solve a linear system (often via CG/fixed-point/HVP routines) to get hypergradients without full unrolling.

Why this idea may still not fully solve our issue alone:

- Our instability is not only "missing solver API"; it is also driven by model-scale curvature heterogeneity, ALM coupling (\(\lambda, \rho\)), and block-specific correction explosions.
- So external solvers help with correctness and cleaner abstractions, but we still need our blockwise/surgical constraints and stability guards.

Practical plan:

- Use TorchOpt/JAXopt-style implicit APIs to cross-check gradients on reduced blocks/toy problems.
- Keep current custom blockwise path for full-scale MUSE runs.
- Treat off-the-shelf solvers as a validation baseline and possible refactor target after stability is settled.

References (verified):

- TorchOpt documentation: https://torchopt.readthedocs.io/en/latest/
- higher repository: https://github.com/facebookresearch/higher
- JAXopt implicit differentiation docs: https://jaxopt.github.io/stable/implicit_diff.html
- Lorraine et al., *Optimizing Millions of Hyperparameters by Implicit Differentiation* (AISTATS 2020): https://proceedings.mlr.press/v108/lorraine20a.html
- Blondel et al., *Efficient and Modular Implicit Differentiation* (arXiv): https://arxiv.org/abs/2105.15183

