# SIBL Implicit-Correction Observations

Use this file as evidence-ready notes for LaTeX.

## Experiment Metadata

- Date: 2026-03-18
- Commit: working tree (uncommitted debug instrumentation)
- GPU/Device: not required for toy solver experiment
- Dataset split: N/A (toy linear-system test)
- Model: N/A (toy)
- Config overrides: N/A (toy)

## Check 1: Paper vs implementation audit

- [x] ALM gradient term includes `rho * r * grad(L_ret)` (fixed by using tensor residual).
- [x] Implicit correction path supports `CG` and `Neumann`.
- [x] Added explicit solver-quality diagnostics (`||Hh-v||/||v||`).

### Additional findings

- `configs/experiment/unlearn/muse/sibl.yaml` currently sets `use_implicit: false`.
  - If unchanged, implicit correction is fully disabled during training.
- Neumann implementation in code previously used a legacy accumulation that does not directly solve `Hh=v`.
  - Added `neumann_variant=richardson` for a proper fixed-point solve.

## Check 2: Toy solver sanity

Source: `debug/results/toy_neumann_vs_cg.md`

- Key finding:
  - Legacy Neumann has substantially worse solve quality than CG and is consistently worse than corrected Richardson-Neumann.
- Numeric evidence:
  - `dim=64, cond=10`: residuals: CG `1.80e-06`, legacy `7.80e-01`, richardson `4.37e-01`.
  - `dim=128, cond=10`: residuals: CG `2.59e-06`, legacy `7.34e-01`, richardson `3.99e-01`.
  - `dim=64, cond=100`: residuals: CG `1.55e-02`, legacy `8.81e-01`, richardson `6.78e-01`.
- Interpretation:
  - For short truncation depth, legacy Neumann is a weak approximation to `H^{-1}v`.
  - Corrected Richardson variant is better but still sensitive to conditioning and iteration budget.
  - For ill-conditioned systems, CG remains the most reliable baseline.

## Check 3: Real training short debug run

Source files:
- `implicit_debug.jsonl`
- `implicit_debug_summary.json`
- dumped vectors (`outer_*_*.npy`)

### Per-step evidence

| outer_iter | solver | variant | linear_residual | rayleigh_ratio | nonpos_count | status |
|---|---|---|---:|---:|---:|---|
|  |  |  |  |  |  |  |

### Notes

- Attempted run command:
  - `DEBUG_SIBL=1 bash scripts/muse_baselines.sh`
- Successful run in correct env:
  - `source /datadrive/miniconda3/etc/profile.d/conda.sh && conda activate unlearning && DEBUG_SIBL=1 bash scripts/muse_baselines.sh`
- Environment blockers encountered/fixed:
  1. missing `hydra` -> installed `hydra-core`
  2. transformers compatibility (`is_torch_tpu_available`) -> added compatibility fallback
  3. optional `deepspeed` import in RMU -> made optional for non-RMU runs
  4. missing `rouge_score` and `sklearn` -> installed
  5. missing `hydra/job_logging/colorlog` -> installed `hydra-colorlog`
  6. missing optional `lm_eval` package -> made evaluator import optional
- Current hard blocker:
  - CUDA OOM during implicit correction vector materialization:
    - failure at `v = g_alm_flat * mask_flat`
    - requested allocation: ~25.1 GiB
    - GPU free at failure: ~7.3 GiB
- Fallback events: N/A (training did not reach outer iterations)
- Residual trend: N/A
- Conditioning proxy trend: N/A
- Any exploding correction norms: N/A

## Check 4: Debug-dir block capture (real model, no full flatten)

Script:
- `debug/capture_implicit_block.py`

Run env:
- `conda activate unlearning`

Artifacts:
- `debug/captured_block/*.npy` + `summary.json`
- `debug/captured_block_stable/*.npy` + `summary.json`

### Findings (run A: default probe alpha)

- Selected block size: `162,529,280` params from layers 30-31.
- Neumann produced unstable values:
  - `alpha ~= 550806.8`
  - `h_norm = inf`, `g_corr_norm = NaN`, `linear_residual = NaN`
- CG was finite but poor solve quality:
  - `linear_residual ~= 9.54`

### Findings (run B: stabilized settings)

Overrides:
- `neumann_mu=1.0`, `neumann_steps=2`, `neumann_alpha_default=0.01`, `neumann_use_probe_alpha=false`
- `cg_damping=0.1`, `cg_iters=20`

Results:
- Neumann became finite:
  - `h_norm ~= 0.452`
  - `g_corr_norm ~= 18.245`
  - `linear_residual ~= 0.972`
- CG became worse in this setup:
  - `linear_residual ~= 216.8`

Interpretation:
- The probe-based alpha is currently too aggressive for this block.
- Fixed small alpha + larger mu stabilizes Neumann numerically.
- CG on this captured block is very sensitive to damping/curvature scaling.

## Check 5: Minimal full training run (no implicit)

Command:
- `conda activate unlearning`
- `python src/train.py ... trainer.method_args.use_implicit=false trainer.method_args.T=1 trainer.method_args.K=1`

Outcome:
- Completed successfully end-to-end (train + eval) on `Llama-2-7b-hf`.
- Confirms the immediate failure mode is specifically implicit full-vector materialization.

## Check 6: Blockwise implicit (layer-by-layer) experiment

Script:
- `debug/blockwise_implicit_experiment.py`

Runs:
- Neumann blockwise: `debug/blockwise_run_neumann/blockwise_report.{json,md}`
- CG blockwise: `debug/blockwise_run_cg/blockwise_report.{json,md}`

Setup:
- selected blocks: last 2 layers (`layer_30`, `layer_31`)
- block dimension per layer: `202,383,360`
- `T=2`, `K=1`

### Neumann (richardson, mu=1.0, steps=2, alpha=0.01)

- Iter 0: `L_ret=0.4795`, `r=+0.3795`, `lambda=0.3795`
- Iter 1: `L_ret=0.3617`, `r=+0.2617`, `lambda=0.6412`
- One block had fallback on iter 1 (`fallback_exploding_correction`), but run remained stable.
- Per-block linear residuals mostly near ~1.0.

### CG (damping=0.1, iters=20)

- Iter 0: `L_ret=0.4797`, `r=+0.3797`, `lambda=0.3797`
- Iter 1: `L_ret=8.0009`, `r=+7.9009`, `lambda=8.2806`
- This reproduces "lambda blowing up quickly" behavior.
- Per-block linear residuals were poor/high (e.g., 41.0, 8.94, 4.97).

Interpretation:
- Blockwise implicit is feasible without OOM.
- CG remains unstable in this setup and can rapidly increase constraint residual.
- Blockwise Neumann with conservative settings is currently the safer option.

## Check 7: Main SIBL port status

Ported to `src/trainer/unlearn/sibl.py`:
- Added blockwise implicit flags:
  - `implicit_blockwise`
  - `implicit_block_last_n_layers`
  - `implicit_block_include_attn`
  - `implicit_block_include_mlp`
- Added layer-block builder and blockwise implicit correction path.
- Kept full implicit path intact for ablations.

Config wiring added in:
- `configs/experiment/unlearn/muse/sibl.yaml`

Current readiness:
- Code compiles and debug blockwise experiments succeed.
- Ready for your full training run with `use_implicit=true` and `implicit_blockwise=true`.

## Check 8: NaN cascade after port (root cause + fix)

Observed log symptom:
- `alpha` from Neumann probe jumped to very large values (e.g., `409200`)
- then `h=inf`, `g_corr=nan`, and losses became `nan`

Root cause:
- Probe Lipschitz estimate used normalized vector `u` but still divided by old `u_norm`, shrinking `L_est` and inflating `alpha`.

Fix applied in `src/trainer/unlearn/sibl.py`:
- Corrected probe estimate to use `L_est = ||H u||` after `u` normalization.
- Added alpha clamp bounds:
  - `neumann_alpha_min`
  - `neumann_alpha_max`
- Added non-finite guard for `v` in Neumann path.
- Added early stop when outer-step metrics become non-finite.

Config safety defaults set in `configs/experiment/unlearn/muse/sibl.yaml`:
- `neumann_use_probe_alpha: false`
- `neumann_alpha_default: 0.01`
- `neumann_alpha_min: 1e-6`
- `neumann_alpha_max: 0.1`
- `neumann_mu: 1.0`
- `neumann_steps: 2`

Short sanity run result (`T=2`, blockwise Neumann):
- No NaN/Inf in implicit logs.
- Iter0: `L_fgt=24.3097`, `L_ret=0.4786`, `lambda=0.379`
- Iter1: `L_fgt=23.4580`, `L_ret=0.3628`, `lambda=0.641`

## Interim decision

- Current best method:
  - Use `implicit_solver=cg` as stability baseline.
  - Use `implicit_solver=neumann` with `neumann_variant=richardson` for ablations.
- Why:
  - Toy evidence shows legacy Neumann underperforms.
  - Corrected Neumann is better but still degrades on high condition numbers.
- Next change to test:
  - Run short real MUSE training with debug enabled (`DEBUG_SIBL=1`) and compare:
    1) `implicit_solver=cg`
    2) `implicit_solver=neumann`, `neumann_variant=richardson`

## Final recommendation

- Keep/drop implicit correction?
- Preferred solver and hyperparameters:
- Remaining risks:
