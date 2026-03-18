# SIBL Implicit-Correction Debug Kit

This folder contains **small, reproducible debug experiments** for diagnosing why implicit correction is not helping in SIBL unlearning.

For a non-jargon, step-by-step explanation of findings and fixes, read:
- `debug/implicit_correction_walkthrough.md`

## What was instrumented

Code changes in `src/trainer/unlearn/sibl.py` add:

- A switchable Neumann implementation:
  - `neumann_variant=legacy` (original behavior)
  - `neumann_variant=richardson` (proper fixed-point solve for `H^{-1}v`)
- Per-outer-step implicit diagnostics:
  - linear solve residual `||Hh - v|| / ||v||`
  - random Rayleigh quotient conditioning proxy
  - norm tracking (`||v||`, `||h||`, `||g_corr||`)
- Optional `.npy` dumps of vectors (`v`, `h`, `g_corr`, etc.) for offline analysis
- Optional early stop after a few outer steps (`debug_stop_after_outer`)

Also fixed an implementation issue in ALM:

- The AL penalty term now uses tensor residual `r_tensor = L_ret - epsilon` (instead of detached `.item()`), so gradients correctly include the `rho * r * grad(L_ret)` component.

## Quick experiments

### 1) Pure toy linear-system check (fast, no model training)

Compares:
- CG
- legacy Neumann
- corrected (Richardson) Neumann

Run:

```bash
python debug/toy_neumann_vs_cg.py
```

Outputs:
- `debug/results/toy_neumann_vs_cg.md`
- `debug/results/toy_neumann_vs_cg.json`

### 2) Short real SIBL debug run (few outer steps)

From repo root:

```bash
DEBUG_SIBL=1 bash scripts/muse_baselines.sh
```

This enables debug overrides only for SIBL:
- `use_implicit=true`
- `debug_implicit=true`
- `debug_save_arrays=true`
- `debug_stop_after_outer=2`
- `T=3`, `K=1`
- `implicit_solver=neumann`
- `neumann_variant=richardson`

### 3) Analyze saved training debug artifacts

After a debug run finishes:

```bash
python debug/analyze_implicit_debug.py --debug-dir <your_output_dir>/debug
```

Example:

```bash
python debug/analyze_implicit_debug.py \
  --debug-dir saves/unlearn/muse_Llama-2-7b-hf_News_SIBL_debug_implicit/debug
```

Outputs:
- `implicit_analysis.md`
- `implicit_analysis.json`

### 4) Capture implicit matrices on a selected block (recommended first)

This avoids full-model flatten OOM and still saves real vectors from one SIBL step.

Run in your conda env:

```bash
source /datadrive/miniconda3/etc/profile.d/conda.sh
conda activate unlearning
PYTHONPATH=src python debug/capture_implicit_block.py
```

Stabilized variant example:

```bash
PYTHONPATH=src python debug/capture_implicit_block.py \
  --out-dir debug/captured_block_stable \
  --neumann-mu 1.0 \
  --neumann-steps 2 \
  --neumann-alpha-default 0.01 \
  --cg-damping 0.1 \
  --cg-iters 20
```

Outputs:
- `g_flat.npy`, `mask_flat.npy`, `v_flat.npy`
- `g_corr_neumann.npy`, `h_cg.npy`
- `summary.json`, `summary.md`

## Suggested decision flow

1. Run toy test first. If `legacy` residual is much worse than `richardson`, legacy Neumann is unreliable.
2. Run short real debug with `richardson`.
3. Inspect:
   - `linear_residual` trend
   - `condition_proxy.rayleigh_ratio`
   - fallback statuses
4. If residuals are high or curvature is poor:
   - increase `neumann_mu`
   - reduce `neumann_steps` and/or smaller `eta_theta`
   - try `implicit_solver=cg` with small `cg_damping`

## Notes for paper/LaTeX write-up

Use `debug/observations.md` as a running log. It is intentionally markdown so you can convert findings into LaTeX tables/figures directly.
