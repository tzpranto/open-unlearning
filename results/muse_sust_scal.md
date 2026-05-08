# MUSE Sustainability & Scalability (Llama-2-7b-hf, News)

Updated: 2026-05-08

## Scalability

Single-shot unlearning on progressively larger forget sets (1x, 2x, 3x, 4x the standard 889 samples).

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain).

| Scale | forget_size | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| scal/forget_1 (=seed42) | 889 | 0.542 | 0.065 | 0.522 | 0.015 | 0.581 |
| scal/forget_2 | 1778 | 0.495 | 0.050 | 0.467 | 0.016 | 0.579 |
| scal/forget_3 | 2667 | 0.538 | 0.024 | 0.488 | 0.015 | 0.573 |
| scal/forget_4 | 3554 | 0.321 | 0.021 | 0.002 | 0.014 | 0.005 |

## Sustainability

Sequential unlearning: each step applies PDU on a new 889-sample forget set, starting from the previous step's checkpoint (5-fold seed=42 model).

HM = harmonic mean of (1-forget_knowmem, 1-verbmem, retain).

| Step | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- |
| step 1 (=seed42) | 0.542 | 0.065 | 0.522 | 0.015 | 0.580 |
| step 2 (forget_2) | 0.554 | 0.007 | 0.470 | 0.008 | 0.558 |
| step 3 (forget_3) | 0.550 | 0.001 | 0.005 | 0.008 | 0.016 |
| step 4 (forget_4) | 0.591 | 0.003 | 0.051 | 0.008 | 0.130 |

## LLM Judge — Scalability

FL = forget leakage (lower = better). RA = retain accuracy, RQ = response quality (higher = better). HM = hmean(1-FL/2, RA/2, ret_RQ/2).

| Scale | FL↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- |
| scal/forget_1 (=seed42) | 0.84 | 1.21 | 1.68 | 0.656 |
| scal/forget_2 | 0.75 | 1.08 | 1.62 | 0.640 |
| scal/forget_3 | 0.74 | 1.22 | 1.72 | 0.683 |
| scal/forget_4 | 0.53 | 0.02 | 0.02 | 0.015 |

## LLM Judge — Sustainability

| Step | FL↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- |
| step 1 (=seed42) | 0.84 | 1.21 | 1.68 | 0.656 |
| step 2 (forget_2) | 0.70 | 1.20 | 1.66 | 0.680 |
| step 3 (forget_3) | 0.70 | 0.00 | 0.00 | 0.000 |
| step 4 (forget_4) | 0.73 | 0.03 | 0.03 | 0.022 |

## BLADE — Scalability

Single-shot unlearning on progressively larger forget sets. Config: eps_mul=3.2, eta_theta=3e-5, K=3, tau=0.7, conv_patience=20.

| Scale | forget_size | T | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scal/forget_1 (=vanilla) | 889 | 300 | 0.550 | 0.173 | 0.475 | 0.031 | 0.542 |
| scal/forget_2 | 1778 | 300 | 0.504 | 0.321 | 0.500 | 0.084 | 0.547 |
| scal/forget_3 | 2667 | 400 | 0.466 | 0.366 | 0.505 | 0.116 | 0.552 |
| scal/forget_4 | 3554 | 500 | 0.505 | 0.386 | 0.504 | 0.137 | 0.533 |

## BLADE — Scalability LLM Judge (Claude Opus 4.7)

| Scale | FL↓ | FL_know↓ | FL_verb↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| scal/forget_1 (=vanilla) | 0.855 | 1.35 | 0.36 | 0.88 | 1.72 | 0.579 |
| scal/forget_2 | 1.010 | 1.22 | 0.80 | 0.90 | 1.63 | 0.549 |
| scal/forget_3 | 1.010 | 1.15 | 0.87 | 0.92 | 1.67 | 0.556 |
| scal/forget_4 | 1.045 | 1.14 | 0.95 | 0.87 | 1.61 | 0.532 |

## BLADE — Sustainability

Sequential unlearning: each step applies BLADE on a new 889-sample forget set, initializing LoRA from previous step's adapters.

| Step | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | HM↑ |
| --- | --- | --- | --- | --- | --- |
| step 1 (=vanilla) | 0.550 | 0.173 | 0.475 | 0.031 | 0.542 |
| step 2 | 0.539 | 0.212 | 0.485 | 0.046 | 0.545 |
| step 3 | 0.556 | 0.230 | 0.459 | 0.056 | 0.523 |
| step 4 | 0.564 | 0.222 | 0.469 | 0.051 | 0.525 |

## BLADE — Sustainability LLM Judge (Claude Opus 4.7)

| Step | FL↓ | FL_know↓ | FL_verb↓ | RA↑ | ret_RQ↑ | HM↑ |
| --- | --- | --- | --- | --- | --- | --- |
| step 1 | 0.94 | 1.24 | 0.64 | 0.89 | 1.67 | 0.563 |
| step 2 | 0.84 | 1.24 | 0.45 | 0.89 | 1.66 | 0.579 |
| step 3 | 0.86 | 1.25 | 0.48 | 0.87 | 1.58 | 0.563 |
| step 4 | 0.84 | 1.28 | 0.40 | 0.92 | 1.62 | 0.584 |

## Notes

- All runs use seed=42, bsz=2, accum=8, gradient_checkpointing=true
- Model: Llama-2-7b-hf, Data: MUSE News
- Gold (retrain): forget_knowmem=0.324, retain=0.552, HM=0.660
- Scalability tests if method handles larger forget sets in one shot
- Sustainability tests if method can be applied repeatedly without destroying the model
- PDU: scal/forget_1 and sust step 1 use the canonical 5-fold seed=42 PDU result directly
- PDU Sustainability v2: steps 2-4 chain from the 5-fold s42 checkpoint (not a fresh run)
- PDU Scalability: retain collapses at 4x scale (3554 samples) but holds at 1-3x
- PDU Sustainability: retain collapses after step 2 (HM 0.58→0.56→0.02→0.13); model destroyed by step 3
- LLM Judge confirms: PDU step 2 holds (HM=0.68) but steps 3-4 produce incoherent outputs (RA=0, RQ=0)
- BLADE Scalability: HM *increases* with data size (0.542→0.547→0.552) — retains better with more data. Verbmem degrades but knowmem improves. LR calibration becomes conservative with large data (fold 4 lr=9e-06).
- BLADE Sustainability: uses LoRA chaining (init from previous fold's adapters)
