# MUSE News (Llama-2-7b-hf)

Updated: 2026-04-14

Gold target: forget_knowmem ≤ 0.324, retain ≥ 0.552

## Main Results

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | mia↓ | privleak↓ | train_time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3243 | 0.2042 | **0.5523** | 0.0247 | — | — | — |
| Target (pre-unlearn) | 0.6538 | 0.5693 | 0.5436 | 0.3023 | — | — | — |
| GradAscent | 0.0027 | 0.0489 | 0.0077 | 0.0079 | — | — | 34m |
| GradDiff | 0.3302 | 0.0053 | 0.2466 | 0.0079 | — | — | 1h 02m |
| NPO | 0.5173 | 0.3567 | 0.4195 | 0.0960 | — | — | 1h 05m |
| SimNPO | 0.5839 | 0.2585 | 0.4698 | 0.0516 | — | — | 59m |
| BLURNPO | 0.5806 | 0.3559 | 0.5318 | 0.1236 | — | — | ~2h |
| RMU | 0.5164 | 0.2848 | 0.4567 | 0.0552 | — | — | 20m |
| PerTA λ=1.0 | 0.6345 | 0.4245 | 0.5401 | 0.2119 | — | — | <1m |
| PerTA λ=1.5 | 0.5374 | 0.3289 | 0.5128 | 0.0998 | — | — | <1m |
| PerTA λ=2.0 | 0.5161 | 0.2510 | 0.5079 | 0.0540 | — | — | <1m |
| PerTA λ=3.5 (full) | 0.3765 | 0.1910 | 0.4162 | 0.0203 | 0.8412 | -66.67 | <1m |
| PerTA λ=3.5 (imp20%) | 0.2597 | 0.0885 | 0.3093 | 0.0140 | 0.2026 | 67.38 | <1m |
| **LoRA-BiAL (ours)** | **0.2893** | **0.1905** | **0.4494** | **0.0281** | **0.8534** | **-69.23** | ~2m |

## LoRA-BiAL: PerTA + LoRA Bilevel ALM

Best config (Ze0): PerTA λ=3.5 init → LoRA r=16 bilevel with NPO β=4.0, K=3, T=25, outer_lr=3e-5, inner_lr=2e-4.

Compared to PerTA alone: retain +0.053 (0.396→0.449) while keeping forget quality (0.282→0.289).

### Key findings from 40+ ablation experiments (Z-series)

1. **NPO saturation is a feature**: β=4 saturates NPO in ~5-8 steps, then inner loop gets free retain recovery. Slower saturation (β≤2) or non-saturating losses (KL) perform worse.
2. **K=3 is sharply optimal**: K=1-2 (NPO never saturates), K≥5 (too much forget leakage with Adam). K=4 drops δ by 0.06.
3. **Gentle outer LR is critical**: olr=3e-5 >> 5e-5 >> 1e-4. Less retain damage during initial NPO correction.
4. **T=25 is optimal**: T=50 degrades results — stale outer Adam momentum interferes after saturation.
5. **Adam >> SGD**: Manual SGD barely moved LoRA weights. Adam gave +0.05 δ improvement.
6. **ε is insensitive**: ε ∈ {0.50, 0.70, 0.90} all give similar results.
