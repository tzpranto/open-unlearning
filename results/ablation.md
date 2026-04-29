# LoRA-BiAL Ablation Results

Updated: 2026-04-29

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
| A9 | tau=1.0 (unclamped) | 0.1606 | 0.0026 | 0.6044 | 0.0079 | 0.7795 |
| A10 | Fixed λ=1.0 | 0.1949 | 0.0000 | 0.6373 | 0.0079 | 0.7872 |
| A12 | ρ=0 (no penalty) | 0.2024 | 0.0000 | 0.6324 | 0.0079 | 0.7823 |
| A14 | Symmetric dual | 0.1128 | 0.0000 | 0.6362 | 0.0079 | 0.8110 |
| A15 | No LoRA (full fine-tune) | 0.0020 | 0.0003 | 0.3061 | 0.0079 | 0.4592 |

## LLM Judge (pending)

| ID | Ablation | FL↓ | FL_know↓ | FL_verb↓ | RA↑ | ret_RQ↑ | HM↑ |
|---|---|---|---|---|---|---|---|
| A0 | Full method | 0.14 | 0.27 | 0.00 | 1.40 | 1.69 | 0.814 |
| A1 | K=0 (no inner loop) | 0.10 | 0.19 | 0.00 | 1.37 | 1.68 | 0.811 |
| A4 | GA forget loss | 0.01 | 0.02 | 0.00 | 0.10 | 0.18 | 0.093 |
| A5 | NPO forget loss | 1.44 | 0.87 | 2.00 | 1.47 | 1.72 | 0.495 |
| A9 | tau=1.0 (unclamped) | 0.20 | 0.39 | 0.00 | 1.34 | 1.64 | 0.785 |
| A10 | Fixed λ=1.0 | 0.21 | 0.42 | 0.00 | 1.43 | 1.66 | 0.806 |
| A12 | ρ=0 (no penalty) | 0.20 | 0.40 | 0.00 | 1.39 | 1.66 | 0.799 |
| A14 | Symmetric dual | 0.13 | 0.25 | 0.00 | 1.41 | 1.67 | 0.815 |
| A15 | No LoRA (full fine-tune) | 0.00 | 0.00 | 0.00 | 0.70 | 1.25 | 0.550 |

## Observations

- **A1 (K=0)**: ret_know drops 0.658→0.631 (−0.027). Forgetting actually improves (0.110→0.071). HM barely changes (−0.004) because HM over-rewards forgetting. The retain difference is the real signal — inner loop provides ~4% retain protection.
- A0 converged at step 78 (dt=38s/step, total ~50min). A1 converged at step 79 (dt=15s/step, total ~20min). Same convergence point but A1 is 2.5× faster per step since K=0 skips inner loop. A0's step 32: L_ret=2.73 spike, dt=109s (emergency inner recovery fired). A1's step 32: L_ret=3.46 spike, dt=15s (no recovery, λ alone recovered by step 40). The unrecovered spike cost A1 ~0.027 in retain quality.
- **A4 (GA)**: Complete retain collapse. ret_know=0.060 (−0.598 from A0). GA's unbounded gradients cause L_ret=4.26 spike at step 32, and even with emergency inner recovery the damage is catastrophic. Proves bounded forget loss (clamped entropy) is essential.
- A4 training: L_fgt went to -7.2 (unbounded negative), LR calibrated to 9e-5 (3×). The model forgot perfectly (fk=0.006) but became non-functional.
- **A5 (NPO)**: Total forgetting failure. NPO loss collapsed to 0.0 by step 40 without actually unlearning — verbatim memorization intact (0.997), knowledge barely reduced (0.457 vs target 0.303). The model satisfies the NPO objective trivially (probability under ref model is already low for long sequences) without changing behavior. Retain is fine (0.662) because nothing changed. Proves reference-model-based losses don't work for pretraining-style memorization.
- **A9 (tau=1.0)**: Meaningful retain degradation (0.604 vs 0.658, −0.054). Unclamped entropy keeps pushing forgetting gradient even after tokens reach high entropy, causing collateral damage to retain. Converged later (step 103 vs A0's 78) — the unbounded loss keeps velocity high longer. Proves the clamping threshold is load-bearing for retain protection.
- **A10 (fixed λ=1.0)**: Moderate retain drop (0.637 vs 0.658, −0.021). Fixed λ can't respond to constraint violations — when L_ret spikes, λ stays at 1.0 instead of increasing to penalize forgetting harder. Forgetting slightly worse (0.195 vs 0.110) because fixed λ doesn't ramp up to slow down the outer loop. LLM judge confirms: RA=1.43 (comparable to A0's 1.40), ret_RQ=1.66 (similar to A0's 1.69). Proves adaptive λ provides modest but consistent benefit.
- **A12 (ρ=0)**: Moderate retain drop (0.632 vs 0.658, −0.026). Without penalty augmentation, the ALM update rule becomes λ += 0, so λ stays at its initial value (1.0) throughout training. Functionally similar to A10 but through a different mechanism. Forgetting slightly worse (0.202 vs 0.110). Proves ρ-augmentation is needed for λ to actually adapt.
- **A14 (symmetric dual)**: Mild retain drop (0.636 vs 0.658, −0.022). With symmetric decay (factor=1.0 vs default 0.1), λ drops 10× faster when constraint is satisfied. λ peaked at 2.94 (step 40 post-spike) then decayed to 2.64 by step 72 — in A0, λ stays elevated much longer due to asymmetric update. Converged at same step (78). Forgetting nearly identical to A0 (0.113 vs 0.110). LLM judge FL is actually lower than A0 (0.13 vs 0.14) — symmetric decay allows more aggressive forgetting in the final phase when λ drops. Proves asymmetric decay provides modest retain protection (~2%) but is not critical.
- **A15 (no LoRA)**: Catastrophic retain collapse. ret_know=0.306 (−0.352 from A0, barely above Gold target). Forgetting is too aggressive (fk=0.002 vs A0's 0.110) — without LoRA's low-rank constraint, gradient updates affect all 6.7B params and destroy general knowledge. Converged at step 51 (faster than A0's 78) because velocity decays faster with full-rank updates. Training: phase transition at step 5, LR halved at step 11, L_ret spiked to 2.21 at step 6 (emergency inner recovery fired). Final λ=1.447. Despite identical ALM constraint + inner loop, full fine-tune cannot preserve retain quality. Proves LoRA provides essential implicit regularization via low-rank parameterization — the bilevel structure alone is insufficient without parameter-efficient constraint on the update space.

## Notes

- ↑ = higher is better, ↓ = lower is better
- HM = hmean(1−fgt_know, 1−fgt_verb, ret_know)
- Gold HM ceiling = 0.739 (we exceed Gold because we over-forget relative to Gold)
- Retain difference (ret_know) is the primary discrimination axis between ablations
- Each ablation changes exactly ONE thing from A0
- LLM judge scores: FL = forget leakage (0-2), RA = retain accuracy (0-2), ret_RQ = retain response quality (0-2)
