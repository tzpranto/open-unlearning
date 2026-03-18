# Conditioning and Sparsity Check (Captured Blocks)

This note answers two specific questions:

1. Are the captured implicit blocks ill-conditioned?
2. Do the captured blocks contain too many zeros?

Data source:
- `debug/captured_block/summary.json`
- `debug/captured_block_stable/summary.json`
- `debug/conditioning_sparsity_stats.json`
- `debug/blockwise_run_neumann/conditioning_rows.json`
- `debug/blockwise_run_cg/conditioning_rows.json`

---

## 1) What "ill-conditioned" means

For a linear solve \(Hh=v\), conditioning is often measured by
\[
\kappa(H) = \frac{\sigma_{\max}(H)}{\sigma_{\min}(H)}
\]

- Large \(\kappa(H)\): small perturbations can cause large solution changes.
- Typical practical interpretation:
  - \(\kappa \approx 1\): very well-conditioned
  - \(\kappa \gg 10^3\): often considered problematic/ill-conditioned (context dependent)

References:
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, SIAM.
- L. N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM.

Important caveat for our debug metric:
- We used a **Rayleigh proxy** from random probes, not full spectrum.
- So this is a heuristic indicator, not an exact \(\kappa(H)\).

---

## 2) Conditioning results from our runs

## 2.1 Captured block (default run, unstable Neumann alpha probe)

- Neumann output had `NaN/Inf` (`h_norm=inf`, `g_corr_norm=NaN`), so conditioning diagnosis from this run is not trustworthy.
- CG linear residual: `9.54` (poor solve quality).

## 2.2 Captured block (stable run, fixed alpha + stronger mu)

- Neumann linear residual: `0.9716` (finite, moderate quality).
- Rayleigh ratio proxy: `1.00000095` (looks very well-conditioned under this proxy).
- CG linear residual: `216.80` (very poor solve quality in this configuration).

## 2.3 Blockwise run inside layer-wise setup

Neumann blockwise:
- layer_30 residual `1.0625` (iter 0), fallback on iter 1
- layer_31 residual `0.9023` (iter 0), `0.9141` (iter 1)

CG blockwise:
- layer_30 residuals: `41.0`, `8.9375`
- layer_31 residuals: `1.9063`, `4.9688`

Practical interpretation:
- We do **not** see evidence of severe ill-conditioning from the Neumann Rayleigh proxy itself.
- But we do see solver instability/poor solves (especially CG) from large residuals.
- So current issue is better described as **solver-quality instability** than "proven highly ill-conditioned Hessian."

---

## 3) Zero-density (sparsity) checks

From `debug/conditioning_sparsity_stats.json`:

Stable captured run (`captured_block_stable`):
- `g_flat_zero_frac`: `0.0000803` (~0.008%)
- `mask_flat_zero_frac`: `0.0277374` (~2.77%)
- `v_flat_zero_frac`: `0.0278150` (~2.78%)
- `g_corr_neumann_zero_frac`: `0.0280778` (~2.81%)
- `h_cg_zero_frac`: `0.0277374` (~2.77%)

Unstable captured run (`captured_block`):
- similar zero fractions for `g_flat/mask/v_flat`
- `g_corr_neumann_nonfinite_frac = 1.0` (all non-finite due unstable alpha), not a sparsity issue

Interpretation:
- Captured blocks are **not dominated by zeros**.
- Zero fraction is small-to-moderate and consistent with mask structure (~2.7% zeros in this captured region).
- Failure was numerical instability (NaN/Inf), not "too sparse / all-zero signal."

---

## 4) Bottom line

- Ill-conditioning (in strict matrix sense) is **not strongly supported** by current proxy numbers.
- The dominant observed issue is **unstable implicit solve quality**, especially with CG and with aggressive Neumann alpha probing.
- Blockwise Neumann with conservative settings is currently the most stable option.

