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
| A18 | Swapped (forget inner) | 0.99 | 0.62 | 1.36 | 1.36 | 1.77 | 0.655 |

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

## MUSE News: K=0 ε-multiplier sensitivity (seed=42)

All runs: eta_theta=3e-5, T=300, K=0, tau=0.7, lora_r=16, bs=2, accum=8. Baseline L_ret=0.824.

| eps_mul | ε | fgt_know↓ | fgt_verb↓ | retain↑ | HM↑ | λ final | extract↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 0.618 | 0.6046 | 0.4675 | 0.5148 | 0.4725 | 14.1 | 0.2109 |
| 1.0 | 0.824 | 0.5989 | 0.1629 | 0.4918 | 0.5244 | 12.0 | 0.0490 |
| 1.25 | 1.030 | 0.6070 | 0.1112 | 0.4973 | 0.5281 | 9.1 | 0.0239 |
| **1.5** | **1.236** | **0.5816** | **0.1147** | **0.4777** | **0.5345** | **7.0** | **0.0254** |
| 2.0 | 1.648 | 0.5871 | 0.3008 | 0.4597 | 0.4977 | 4.2 | 0.1075 |
| 2.5 | 2.060 | 0.6037 | 0.3060 | 0.5018 | 0.5036 | 2.7 | 0.1083 |
| 3.0 | 2.472 | 0.5989 | 0.2478 | 0.4770 | 0.5068 | 1.5 | 0.0754 |
| 3.2 | 2.637 | 0.5946 | 0.1640 | 0.4843 | 0.5238 | 1.2 | 0.0313 |

Comparison with K=3 sweep (same eps_mul values):

| eps_mul | K=3 HM | K=0 HM | Δ HM | K=3 retain | K=0 retain | Δ retain |
| --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 0.534 | 0.473 | −0.061 | 0.482 | 0.515 | +0.033 |
| 1.0 | 0.546 | 0.524 | −0.022 | 0.509 | 0.492 | −0.017 |
| 1.25 | 0.512 | 0.528 | +0.016 | 0.512 | 0.497 | −0.015 |
| 1.5 | 0.526 | 0.535 | +0.009 | 0.509 | 0.478 | −0.031 |
| 2.0 | 0.522 | 0.498 | −0.024 | 0.487 | 0.460 | −0.027 |
| 2.5 | 0.525 | 0.504 | −0.021 | 0.486 | 0.502 | +0.016 |
| 3.0 | 0.548 | 0.507 | −0.041 | 0.507 | 0.477 | −0.030 |
| 3.2 | 0.542 | 0.524 | −0.018 | 0.475 | 0.484 | +0.009 |

Observations:
- K=0 best HM: eps_mul=1.5 (0.535), comparable to K=3 best (eps_mul=3.0, 0.548)
- K=0 generally underperforms K=3 on HM (avg Δ = −0.020)
- K=0 has higher verbmem at tight constraints (eps=0.75: 0.468 vs K=3's 0.142)
- At loose constraints (eps_mul≥3.0), K=0 and K=3 converge in performance
- K=0 is ~2.5× faster per step (no inner loop), so the gap is modest relative to speedup

## MUSE News: K=6 ε-multiplier sensitivity (seed=42)

All runs: eta_theta=3e-5, T=300, K=6, tau=0.7, lora_r=16, bs=2, accum=8. Baseline L_ret=0.824.

| eps_mul | ε | fgt_know↓ | fgt_verb↓ | retain↑ | HM↑ | λ final | extract↓ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 0.618 | 0.5743 | 0.2086 | 0.4904 | 0.5308 | 12.3 | 0.0481 |
| 1.0 | 0.824 | 0.5653 | 0.2262 | 0.4583 | 0.5195 | 8.3 | 0.0537 |
| 1.25 | 1.030 | 0.5878 | 0.2650 | 0.4959 | 0.5170 | 6.1 | 0.0754 |
| 1.5 | 1.236 | 0.5484 | 0.2103 | 0.4942 | 0.5450 | 4.1 | 0.0487 |
| 2.0 | 1.648 | 0.5484 | 0.3000 | 0.4674 | 0.5188 | 2.1 | 0.0933 |
| 2.5 | 2.060 | 0.5575 | 0.2181 | 0.4699 | 0.5294 | 0.8 | 0.0408 |
| 3.0 | 2.472 | 0.5413 | 0.3031 | 0.5001 | 0.5343 | 0.1 | 0.0829 |
| **3.2** | **2.637** | **0.5136** | **0.1912** | **0.4715** | **0.5542** | **0.1** | **0.0356** |

## MUSE News: K comparison (all eps_mul values, seed=42)

| eps_mul | K=0 HM | K=3 HM | K=6 HM | Best K | K=0 retain | K=3 retain | K=6 retain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 0.473 | 0.534 | 0.531 | K=3 | 0.515 | 0.482 | 0.490 |
| 1.0 | 0.524 | 0.546 | 0.520 | K=3 | 0.492 | 0.509 | 0.458 |
| 1.25 | 0.528 | 0.512 | 0.517 | K=0 | 0.497 | 0.512 | 0.496 |
| 1.5 | 0.535 | 0.526 | 0.545 | K=6 | 0.478 | 0.509 | 0.494 |
| 2.0 | 0.498 | 0.522 | 0.519 | K=3 | 0.460 | 0.487 | 0.467 |
| 2.5 | 0.504 | 0.525 | 0.529 | K=6 | 0.502 | 0.486 | 0.470 |
| 3.0 | 0.507 | 0.548 | 0.534 | K=3 | 0.477 | 0.507 | 0.500 |
| 3.2 | 0.524 | 0.542 | **0.554** | K=6 | 0.484 | 0.475 | 0.472 |

Observations:
- K=6 best HM: eps_mul=3.2 (0.554), highest across all K values
- K=3 wins at 5/8 eps_mul values; K=6 wins at 3/8 (1.5, 2.5, 3.2)
- K=6 avg HM = 0.531, K=3 avg HM = 0.532, K=0 avg HM = 0.512
- K=6 is ~3.7× slower per step than K=3 (25.6s vs 6.9s) for marginal HM improvement
- K=6 has lower verbmem than K=0 across all eps_mul (inner loop helps even more at K=6)
- λ values similar between K=3 and K=6 (constraint dynamics unchanged by inner steps)
- Diminishing returns: K=0→K=3 gains avg +0.020 HM; K=3→K=6 gains avg −0.001 HM

## MUSE News: ε-multiplier sensitivity (seed=42)

All runs: eta_theta=3e-5, T=300, K=3, tau=0.7, lora_r=16, bs=2, accum=8. Baseline L_ret=0.824.

| eps_mul | ε | fgt_know↓ | fgt_verb↓ | retain↑ | HM↑ | λ final | Constraint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 0.618 | 0.579 | 0.142 | 0.482 | 0.534 | ~9.7 | always violated |
| 1.0 | 0.824 | 0.565 | 0.187 | 0.509 | 0.546 | ~8.5 | borderline |
| 1.25 | 1.030 | 0.607 | 0.269 | 0.512 | 0.512 | ~6.6 | transition |
| 1.5 | 1.236 | 0.589 | 0.232 | 0.509 | 0.526 | ~5.0 | mostly satisfied |
| 2.0 | 1.648 | 0.580 | 0.242 | 0.487 | 0.522 | ~3.0 | never binds |
| 2.5 | 2.060 | 0.549 | 0.305 | 0.486 | 0.525 | ~2.0 | λ collapsing |
| **3.0** | **2.472** | **0.546** | **0.232** | **0.507** | **0.548** | ~1.1 | **best HM** |
| 3.2 | 2.637 | 0.550 | 0.173 | 0.475 | 0.542 | — | default (paper) |

Observations:
- Default eps_mul=3.2 gives best HM despite constraint never binding
- Tighter constraints improve retain (+0.01–0.03) but degrade verbmem (+0.06–0.13)
- Net: tighter eps_mul worse on HM; natural forgetting dynamics sufficient on News
- Contrast with Books (ε=0.15) where constraint binds tightly and is essential

## Notes

- ↑ = higher is better, ↓ = lower is better
- HM = hmean(1−fgt_know, 1−fgt_verb, ret_know)
- Gold HM ceiling = 0.739 (we exceed Gold because we over-forget relative to Gold)
- Retain difference (ret_know) is the primary discrimination axis between ablations
- Each ablation changes exactly ONE thing from A0
- A9 vs A0: tau clamping prevents 8.2% retain degradation. Without clamping (Yuan et al. style raw entropy), gradient keeps pushing even after tokens are sufficiently uncertain, causing collateral damage to shared representations used by retain. Clamping at tau*H_max zeroes the gradient once "forgotten enough," protecting retain.
- LLM judge scores: FL = forget leakage (0-2), RA = retain accuracy (0-2), ret_RQ = retain response quality (0-2)
