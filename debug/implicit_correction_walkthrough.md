# Implicit Correction Debug Walkthrough (Team-Friendly)

This note explains, step by step, how the implicit-correction issue was diagnosed, what was tested, and what currently works.

Audience:
- Team members with mixed ML background
- People comfortable with math, but not necessarily with deep learning internals

---

## 1) Problem statement in plain words

We use SIBL unlearning with an optional **implicit correction** step.

When implicit correction was enabled, training failed with out-of-memory (OOM), even when normal training (without implicit correction) could run.

Goal:
- Understand **why** implicit correction fails
- Build a debug workflow that is cheap to run
- Save matrices/vectors so we can analyze without re-running full training

---

## 2) What we observed first

### Symptom
- With implicit enabled, training crashed on CUDA OOM at the line building a very large flattened vector:
  - `v = g_alm_flat * mask_flat`

### Key clue
- The crash happened specifically in implicit path, not in standard training.
- A minimal run with `use_implicit=false`, `T=1`, `K=1` completed successfully.

Interpretation:
- The root issue is not "model too big to train at all".
- The issue is "implicit implementation builds large vectors/second-order ops in a memory-heavy way".

---

## 3) What we changed in debug workflow (not "fancy", just practical)

We created a debug-only script:
- `debug/capture_implicit_block.py`

What it does:
1. Runs one real SIBL step (same model/data family)
2. Captures only a selected block of parameters (not full model)
3. Saves vectors/matrices to `.npy` files
4. Writes readable summary in JSON + markdown

Saved artifacts (example):
- `debug/captured_block_stable/g_flat.npy`
- `debug/captured_block_stable/mask_flat.npy`
- `debug/captured_block_stable/v_flat.npy`
- `debug/captured_block_stable/g_corr_neumann.npy`
- `debug/captured_block_stable/h_cg.npy`
- `debug/captured_block_stable/summary.json`

This gives reproducible evidence without repeatedly paying full training cost.

---

## 4) Background: what CG and Neumann are doing here

In implicit correction, we need to approximately solve:

\[
H h = v
\]

- \(H\): Hessian-like curvature operator from inner objective
- \(v\): outer-gradient-related vector
- \(h\): correction direction used by implicit method

Directly inverting \(H\) is too expensive, so we use iterative approximations.

### 4.1 Conjugate Gradient (CG) intuition

CG is an iterative solver for systems like \(Hh=v\) (best when \(H\) is symmetric positive definite or close enough).

Important parameters:
- `cg_iters`: maximum number of CG iterations
  - higher = potentially more accurate, slower
- `cg_tol`: stopping tolerance on residual
  - lower = stricter solve, can be unstable/slow if curvature is bad
- `cg_damping`: adds \(\lambda I\) to stabilize
  - effective system becomes \((H+\lambda I)h=v\)
  - larger damping improves stability but can bias solution

Practical interpretation:
- If residual stays large even with reasonable `cg_iters`, conditioning is likely poor.

### 4.2 Neumann / Richardson intuition

Neumann-style update approximates inverse action using repeated updates instead of full solve.

In our stable variant, this is Richardson-style fixed point:

\[
h_{k+1} = h_k + \alpha (v - H h_k)
\]

Important parameters:
- `neumann_steps`: number of update steps
  - more steps = more refinement (but more compute, possible instability)
- `neumann_mu`: extra damping inside Hessian operator
  - larger value improves numerical stability
- `neumann_alpha_default`: step size \(\alpha\)
  - too large can explode
  - too small can under-correct
- `neumann_use_probe_alpha`:
  - if true: estimate \(\alpha\) automatically from curvature probe
  - if false: use fixed `neumann_alpha_default`
  - probe can be unstable if curvature estimate is noisy

---

## 5) Experiments and evidence

## 5.1 Block capture run A (default-ish settings)

From `debug/captured_block/summary.json`:
- selected dim: `162,529,280`
- Neumann:
  - `alpha ≈ 550,806.8`
  - `h_norm = inf`
  - `g_corr_norm = NaN`
  - `linear_residual = NaN`
- CG:
  - `linear_residual ≈ 9.54` (finite, but poor)

Conclusion:
- Auto-probed alpha was far too aggressive for this block.
- Numerical explosion confirmed.

## 5.2 Block capture run B (stabilized settings)

Settings:
- `neumann_mu=1.0`
- `neumann_steps=2`
- `neumann_alpha_default=0.01`
- `neumann_use_probe_alpha=false`
- `cg_damping=0.1`, `cg_iters=20`

From `debug/captured_block_stable/summary.json`:
- Neumann became finite:
  - `h_norm ≈ 0.452`
  - `g_corr_norm ≈ 18.245`
  - `linear_residual ≈ 0.972`
- CG became much worse in this setup:
  - `linear_residual ≈ 216.8`

Conclusion:
- Neumann can be stabilized with conservative fixed alpha + stronger damping.
- CG is very sensitive here and not currently the better default for this block.

## 5.3 Minimal real training sanity run

`use_implicit=false`, `T=1`, `K=1` completed end-to-end (train + eval), confirming base training path is healthy.

---

## 6) How the issue was identified (timeline)

1. **Observed failure** only when implicit was on
2. **Verified baseline** (implicit off) works
3. **Moved to debug-only script** to avoid full-run cost
4. **Captured real vectors** on parameter block
5. **Compared unstable vs stabilized Neumann settings**
6. **Confirmed numeric stability can be recovered** in block mode

This isolates the issue to:
- memory-heavy full flatten in implicit path
- unstable hyperparameters for Neumann when probe alpha is too aggressive

---

## 7) What is "solved" vs "not yet solved"

### Solved
- We now have a repeatable, low-cost debug pipeline.
- We can save and inspect real implicit vectors/matrices.
- We identified a stable Neumann configuration (at least on captured block).

### Not fully solved yet
- Full-model implicit training still needs a memory-safe implementation (avoid giant flatten allocations).
- After that implementation, we must validate on full training (not only block capture).

---

## 8) Recommended next implementation step

Before merging any major change into main path:

1. Keep using debug capture as gate:
   - verify finite values
   - verify residual is reasonable
2. Implement blockwise/memory-safe implicit update in main SIBL:
   - avoid full `g_alm_flat`/`mask_flat` allocation
3. Re-run short full training with implicit enabled:
   - `T=1..3`, `K=1`
4. Compare against non-implicit baseline:
   - stability first, performance second

---

## 9) Quick glossary

- **Residual**: how far current solution is from satisfying \(Hh=v\), often \(\|Hh-v\|/\|v\|\)
- **Conditioning**: how sensitive the system is to small perturbations
- **Damping**: adding stability term (like \(\lambda I\))
- **HVP**: Hessian-vector product, used to avoid forming full Hessian matrix

---

## 10) One-line summary

The implicit issue was traced to a memory-heavy full-vector implementation plus unstable Neumann step sizing; blockwise debug capture proved a stable setting exists, and the next step is a memory-safe main implementation using the same validated logic.

