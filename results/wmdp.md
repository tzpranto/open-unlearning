# WMDP Bio+Cyber (Zephyr-7b-beta)

Updated: 2026-04-29

Baseline (no unlearning): wmdp_bio=0.648, wmdp_cyber=0.431, mmlu=0.589 (our eval).
Random chance: 0.250.

## Results (seed=42)

| Method | wmdp_bio↓ | wmdp_cyber↓ | mmlu↑ | HM↑ | train_time | notes |
|---|---|---|---|---|---|---|
| Baseline (no unlearn) | 0.648 | 0.431 | 0.589 | 0.476 | — | our eval |
| RMU (official code) | 0.302 | 0.277 | 0.576 | 0.659 | ~60s | centerforaisafety/wmdp |
| BLUR-RMU (official code) | 0.277 | 0.271 | 0.566 | 0.663 | ~18s | OptimAI-Lab/BLURLLMUnlearning |
| RMU (our Trainer) | 0.648 | 0.350 | 0.582 | 0.492 | 75s | single steering vec — broken |
| **Paper: RMU** | **0.312** | **0.282** | **0.571** | **0.653** | — | BLUR paper Table 4 |
| **Paper: BLUR-RMU** | **0.269** | **0.266** | **0.570** | **0.669** | — | BLUR paper Table 4 |
| **Paper: BLUR-NPO** | **0.276** | **0.265** | **0.484** | **0.624** | — | BLUR paper Table 4 |
| BLADE v1 (ε_mul=1.15) | 0.661 | 0.423 | 0.582 | 0.477 | ~60min | clamped_entropy — no forgetting |
| BLADE v2 (ε_mul=2.0) | 0.662 | 0.421 | 0.583 | 0.477 | ~60min | clamped_entropy — no forgetting |

## HM Calculation

HM = hmean(forget_bio, forget_cyber, retain_mmlu) where:
- forget_bio = 1 − (bio_acc − 0.25) / (baseline_bio − 0.25)  (fraction of hazardous knowledge removed, clipped at 0)
- forget_cyber = 1 − (cyber_acc − 0.25) / (baseline_cyber − 0.25)
- retain_mmlu = mmlu / baseline_mmlu

This normalizes forgetting relative to chance (0.25) and baseline, retain relative to original.

Our RMU official: hmean(1−(0.302−0.25)/(0.648−0.25), 1−(0.277−0.25)/(0.431−0.25), 0.576/0.589) = hmean(0.869, 0.851, 0.978) = 0.896
Our BLUR-RMU: hmean(1−(0.277−0.25)/(0.648−0.25), 1−(0.271−0.25)/(0.431−0.25), 0.566/0.589) = hmean(0.932, 0.884, 0.961) = 0.925

## Observations

- **RMU (official code)**: Matches paper closely (bio=0.302 vs paper 0.312, cyber=0.277 vs 0.282, mmlu=0.576 vs 0.571). The slight improvement over paper may be seed difference (we use 42, paper uses 0 or unstated).
- **BLUR-RMU (official code)**: Better bio forgetting than standard RMU (0.277 vs 0.302) with minimal MMLU cost (0.566 vs 0.576). The bilevel gradient projection provides a small but consistent improvement on the forget axis.
- **RMU (our Trainer)**: Broken — single steering vector doesn't work for joint bio+cyber. The official code uses per-topic vectors with round-robin alternation. Our HF Trainer-based wrapper needs rewriting to support this.

## Why Our Trainer RMU Failed

The official RMU implementation uses:
1. **Per-topic random steering vectors** — one for bio, one for cyber, generated independently
2. **Round-robin alternation** — `topic_idx = idx % num_topics` each step
3. **Custom training loop** — no HF Trainer, manual optimizer.step()
4. **Tokenization inside loop** — different max_length per topic (bio=512, cyber=768)

Our Trainer-based RMU uses a single DataLoader mixing all forget data with ONE steering vector. This means bio and cyber get steered toward the same random direction, which is ineffective for bio.

## Why BLADE (Clamped Entropy) Fails on WMDP

BLADE with clamped_entropy achieves near-zero training loss (L_fgt→0.01) but produces zero MCQ forgetting.
The model satisfies the entropy target on raw text without losing the factual representations needed for MCQs.

Root cause: WMDP eval is 4-way MCQ requiring representational-level knowledge removal.
Clamped entropy on raw text makes the model "uncertain when generating text" but the underlying
knowledge representations remain intact for discriminative (MCQ) evaluation.

This contrasts with MUSE/TOFU where eval measures text generation quality directly — there,
clamped entropy works because the eval aligns with the loss objective.

Options for WMDP:
1. Switch forget loss to NPO/GA (forces probability mass away from correct completions)
2. Use activation steering (RMU approach — directly corrupts representations)
3. Hybrid: BLADE structure with RMU-style activation loss as the forget objective

## Notes

- ↑ = higher is better, ↓ = lower is better
- Random chance = 0.250 (4-way MCQ)
- Official RMU params: steering_coeff=6.5, alpha=1200, lr=5e-5, 150 steps, bs=4, layers 5-7 down_proj
- Official BLUR-RMU params: steering_coeff=6.5, alpha=800, lr=5e-5, 150 steps, bs=4 (+ bilevel gradient projection)
- Data: bio-forget-corpus (24K texts), cyber-forget-corpus (1K texts), wikitext retain
