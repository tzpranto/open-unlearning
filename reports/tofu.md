# TOFU (Llama-3.2-1B-Instruct)

Updated: 2026-04-15

Model: open-unlearning/tofu_Llama-3.2-1B-Instruct_full

Metrics: FQ = forget_quality (KS p-value, higher=better; log10 in parentheses), MU = model_utility, ES(Df) = extraction on forget (lower=better), ES(Dr) = extraction on retain (higher=better)

## Forget 1% (40 samples, retain=3960)

Baselines from PerTA paper (Table 2).

| Method | FQ↑ (log10) | MU↑ | ES(Df)↓ | ES(Dr)↑ | notes |
| --- | --- | --- | --- | --- | --- |
| Full (no unlearn) | 0.0068 (-2.17) | 0.599 | 0.743 | 0.737 | |
| GT (retrain) | 1.0000 (0.00) | 0.599 | 0.069 | 0.751 | |
| GA | 0.0111 (-1.95) | 0.597 | 0.189 | 0.656 | |
| GD | 0.0143 (-1.85) | 0.581 | 0.169 | 0.562 | |
| NPO | 0.0087 (-2.06) | 0.595 | 0.178 | 0.650 | |
| NPO+ | 0.0143 (-1.85) | 0.596 | 0.174 | 0.656 | |
| TV | 0.4046 (-0.39) | 0.556 | 0.081 | 0.358 | |
| PerTA-grad | 0.5140 (-0.29) | 0.581 | 0.075 | 0.551 | |
| PerTA-fisher | 0.2655 (-0.58) | 0.586 | 0.085 | 0.600 | |

### v1: Phase 1 (logit_margin, K=1, LR=5e-4, unmasked)

bs=12, eta_theta=5e-4, K=1, implicit=off. 40 samples / bs=12 = 4 steps/epoch.
Bug: logit_margin applied to ALL tokens (prompt + answer). Fixed in v2.

| Checkpoint | FQ↑ (log10) | MU↑ | ES(Df)↓ | fQA_ROUGE↓ | privleak↑ |
| --- | --- | --- | --- | --- | --- |
| step-4 (ep1) | 0.0541 (-1.27) | 0.573 | 0.120 | 0.232 | -9.9 |
| step-8 (ep2) | 0.0541 (-1.27) | 0.017 | 0.029 | 0.007 | 88.9 |

### v2: Phase 1 (logit_margin answer-masked, K=3, LR=2e-4)

bs=12, eta_theta=2e-4, K=3, implicit=off. Answer-token-only logit_margin.
K=3 inner retain steps per outer step = much stronger MU protection.

| Checkpoint | FQ↑ (log10) | MU↑ | ES(Df)↓ | fQA_ROUGE↓ | privleak↑ |
| --- | --- | --- | --- | --- | --- |
| step-2 | 0.0068 (-2.17) | 0.594 | 0.426 | 0.614 | -97.9 |
| step-4 (ep1) | 0.0143 (-1.85) | 0.593 | 0.173 | 0.351 | -44.6 |
| step-6 | 0.0541 (-1.27) | 0.586 | 0.088 | 0.281 | 14.3 |
| **step-8 (ep2)** | **0.4046 (-0.39)** | **0.581** | **0.041** | **0.219** | **68.7** |
| step-10 | 0.2657 (-0.58) | 0.592 | 0.033 | 0.102 | 85.2 |

### Key observations

- **v2 step-8 is a breakthrough**: FQ=0.405 matches TV (0.405) and approaches PerTA-grad (0.514), with MU=0.581 matching GT (0.599)
- **Answer-token masking was critical**: v1 logit_margin on all tokens diluted the signal; v2 concentrates on answer tokens only
- **K=3 prevents MU collapse**: v1 K=1 collapsed MU from 0.573→0.017 at step-8; v2 K=3 holds MU at 0.581
- ES(Df)=0.041 is below GT (0.069) — slight over-forgetting but FQ stays high because truth_ratio distribution is close to retain
- step-10 over-forgets slightly (FQ drops to 0.266) — step-8 is the sweet spot
### v2: Phase 2 (logit_margin + implicit, from step-8)

Fresh LoRA on step-8 merged model. K=3, eta_theta=1e-4, implicit ON (warmup=4).

| Checkpoint | FQ↑ (log10) | MU↑ | ES(Df)↓ | fQA_ROUGE↓ | privleak↑ |
| --- | --- | --- | --- | --- | --- |
| P2 final (step-20) | 0.2657 (-0.58) | 0.585 | 0.125 | 0.279 | -15.5 |

Phase 2 pulled FQ back from 0.405→0.266 — implicit correction recovers retain too aggressively,
undoing some forgetting. **Phase 1 step-8 remains best: FQ=0.405, MU=0.581.**

## Next steps

- Try Phase 2 with higher outer LR or fewer implicit steps
- Run baselines when GPU is free

## Notes

- FQ: KS test p-value comparing forget vs retain truth_ratio distributions. Higher = better (1.0 = perfect match). log10 in parentheses for PerTA comparison.
- Our method: 2-phase (logit_margin cold-start + logit_margin bilevel with implicit correction)
- Model: Llama-3.2-1B-Instruct, LoRA r=16, bilevel with FD-HVP implicit correction
- Key fix: logit_margin masked to answer tokens only (labels != -100)
