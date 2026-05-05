# LoRA-BiAL Ablation Results

Updated: 2026-05-03

## Setup

- **Benchmark**: MUSE Books, Llama-2-7b-hf
- **Base config**: `experiment=unlearn/muse/lora_bial_adaptive_books.yaml`
- **A0 params**: eta_theta=3e-5, T=250, eps_mul=3.2, K=3, tau=0.7, rho=0.1, lambda_init=1.0, conv_patience=20, lora_r=16
- **Gold target**: forget_knowmem ≤ 0.303, retain ≥ 0.687

## Results (seed=42)

| ID | Ablation | fgt_know↓ | fgt_verb↓ | ret_know↑ | extract↓ | HM↑ |
|---|---|---|---|---|---|---|
| **A0** | **Full method (control)** | **0.1101** | **0.0000** | **0.6580** | **0.0079** | **0.8234** |
| A1 | K=0 (no inner loop) | 0.0715 | 0.0000 | 0.6308 | 0.0079 | 0.8192 |
| A4 | GA forget loss | 0.0055 | 0.0032 | 0.0599 | 0.0082 | 0.1604 |
| A5 | NPO forget loss | 0.4567 | 0.9970 | 0.6617 | 0.9160 | 0.0089 |
| A9 | tau=1.0 (unclamped entropy) | 0.1606 | 0.0026 | 0.6044 | 0.0079 | 0.7795 |
| A15 | No LoRA (full fine-tune) | 0.0020 | 0.0003 | 0.3061 | 0.0079 | 0.4592 |
| **A16** | **ALM-off (λ=0, ρ=0)** | **0.0000** | **0.0024** | **0.0919** | **0.0084** | **0.2329** |
| A17 | logit_margin forget loss | 0.0000 | 0.0000 | 0.0000 | 0.0079 | 0.0000 |
| A18 | Swapped (inner=forget, outer=retain) | 0.3720 | 0.6425 | 0.6511 | 0.5477 | 0.5063 |

## LLM Judge (pending)

| ID | Ablation | FL↓ | FL_know↓ | FL_verb↓ | RA↑ | ret_RQ↑ | HM↑ |
|---|---|---|---|---|---|---|---|
| A0 | Full method | 0.14 | 0.27 | 0.00 | 1.40 | 1.69 | 0.814 |
| A1 | K=0 (no inner loop) | 0.10 | 0.19 | 0.00 | 1.37 | 1.68 | 0.811 |
| A4 | GA forget loss | 0.01 | 0.02 | 0.00 | 0.10 | 0.18 | 0.093 |
| A5 | NPO forget loss | 1.44 | 0.87 | 2.00 | 1.47 | 1.72 | 0.495 |
| A9 | tau=1.0 (unclamped entropy) | 0.20 | 0.39 | 0.00 | 1.34 | 1.64 | 0.785 |
| A15 | No LoRA (full fine-tune) | 0.00 | 0.00 | 0.00 | 0.70 | 1.25 | 0.550 |
| A16 | ALM-off (λ=0, ρ=0) | 0.01 | 0.01 | 0.00 | 0.24 | 0.28 | 0.117 |
| A17 | logit_margin forget loss | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |

## Observations

- **A1 (K=0)**: ret_know drops 0.658→0.631 (−0.027). Forgetting actually improves (0.110→0.071). HM barely changes (−0.004) because HM over-rewards forgetting. The retain difference is the real signal — inner loop provides ~4% retain protection.
- A0 converged at step 78 (dt=38s/step, total ~50min). A1 converged at step 79 (dt=15s/step, total ~20min). Same convergence point but A1 is 2.5× faster per step since K=0 skips inner loop. A0's step 32: L_ret=2.73 spike, dt=109s (emergency inner recovery fired). A1's step 32: L_ret=3.46 spike, dt=15s (no recovery, λ alone recovered by step 40). The unrecovered spike cost A1 ~0.027 in retain quality.
- **A4 (GA)**: Complete retain collapse. ret_know=0.060 (−0.598 from A0). GA's unbounded gradients cause L_ret=4.26 spike at step 32, and even with emergency inner recovery the damage is catastrophic. Proves bounded forget loss (clamped entropy) is essential.
- A4 training: L_fgt went to -7.2 (unbounded negative), LR calibrated to 9e-5 (3×). The model forgot perfectly (fk=0.006) but became non-functional.
- **A5 (NPO)**: Total forgetting failure. NPO loss collapsed to 0.0 by step 40 without actually unlearning — verbatim memorization intact (0.997), knowledge barely reduced (0.457 vs target 0.303). The model satisfies the NPO objective trivially (probability under ref model is already low for long sequences) without changing behavior. Retain is fine (0.662) because nothing changed. Proves reference-model-based losses don't work for pretraining-style memorization.
- **A16 (ALM-off)**: Complete system failure. ret_know=0.092 (−0.566 from A0), forget perfect (0.000). With λ=0 and ρ=0, the outer loss is pure clamped_entropy with no retain constraint. The model collapsed at step 31 (L_ret=10.56 triggered safety break). Proves ALM is the core mechanism: without it, bilevel + clamped_entropy alone destroys the model.
- **A17 (logit_margin)**: Total destruction — worst ablation. All metrics zero. Logit margin loss (max_logit - mean_logit) has no natural saturation point unlike clamped entropy. It keeps pushing the model toward uniform logits indefinitely, destroying all representations. Despite ALM constraining retain, the forget gradient is too destructive. Converged at step 200 but model is non-functional. Proves clamped entropy's bounded, self-saturating property is essential — alternative unbounded losses destroy the model even with full ALM protection.
- **A18 (Swapped)**: Complete forgetting failure. fgt_know=0.372 (barely below unlearned model's 0.44), verbatim fully intact (0.643), extraction high (0.548). λ exploded to 367 because the constraint ε_fgt=22.4 exceeds clamped entropy's theoretical max (~8.3) — physically impossible to satisfy. The outer loop successfully minimizes retain (0.651, on par with A0) but the inner loop's "push toward forgetting" via clamped entropy is ineffective: SGD on a bounded loss saturates quickly and the outer ALM penalty overwhelms any forgetting signal. Proves the bilevel direction is load-bearing: inner=retain (immediate spike recovery) + outer=forget (ALM-constrained) is the only viable assignment.
- **A15 (no LoRA)**: Catastrophic retain collapse. ret_know=0.306 (−0.352 from A0, barely above Gold target). Forgetting is too aggressive (fk=0.002 vs A0's 0.110) — without LoRA's low-rank constraint, gradient updates affect all 6.7B params and destroy general knowledge. Converged at step 51 (faster than A0's 78) because velocity decays faster with full-rank updates. Training: phase transition at step 5, LR halved at step 11, L_ret spiked to 2.21 at step 6 (emergency inner recovery fired). Final λ=1.447. Despite identical ALM constraint + inner loop, full fine-tune cannot preserve retain quality. Proves LoRA provides essential implicit regularization via low-rank parameterization — the bilevel structure alone is insufficient without parameter-efficient constraint on the update space.

## MUSE News: K=0 ablation (seed=42) — INVALID, NEEDS RERUN

**WARNING**: This ablation used wrong config (News v2: eps_mul=1.3, eta_theta=2e-5, T=150, ε=0.849) while vanilla champion used (eps_mul=3.2, eta_theta=3e-5, T=300, ε=2.637). Not a valid single-variable ablation. Must rerun with correct config.

| Setting | fgt_know↓ | fgt_verb↓ | ret_know↑ | extract↓ | HM↑ |
|---|---|---|---|---|---|
| **K=3 (full BLADE)** | **0.550** | **0.173** | 0.475 | **0.031** | **0.542** |
| K=0 (no inner loop, WRONG CONFIG) | 0.563 | 0.389 | **0.515** | 0.161 | 0.511 |

## Notes

- ↑ = higher is better, ↓ = lower is better
- HM = hmean(1−fgt_know, 1−fgt_verb, ret_know)
- Gold HM ceiling = 0.739 (we exceed Gold because we over-forget relative to Gold)
- Retain difference (ret_know) is the primary discrimination axis between ablations
- Each ablation changes exactly ONE thing from A0
- A9 vs A0: tau clamping prevents 8.2% retain degradation. Without clamping (Yuan et al. style raw entropy), gradient keeps pushing even after tokens are sufficiently uncertain, causing collateral damage to shared representations used by retain. Clamping at tau*H_max zeroes the gradient once "forgotten enough," protecting retain.
- LLM judge scores: FL = forget leakage (0-2), RA = retain accuracy (0-2), ret_RQ = retain response quality (0-2)
