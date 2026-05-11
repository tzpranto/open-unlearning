# GPU Budget Breakdown

Single NVIDIA H100 (95GB), Azure Standard_NC40ads_H100_v5

## BLADE Training (5-fold, all benchmarks)

| Experiment | Runs | Time/run | GPU-hours |
|---|---|---|---|
| TOFU 1B (3 splits × 5 seeds) | 15 | 12 min | 3.0 |
| TOFU 3B (3 splits × 5 seeds) | 15 | 38 min | 9.5 |
| MUSE Books (5 seeds) | 5 | 104 min | 8.7 |
| MUSE News (5 seeds) | 5 | 120 min | 10.0 |
| KnowUnDo (2 domains × 5 seeds) | 10 | 55 min | 9.2 |
| **Subtotal** | | | **40.4** |

## Baseline Training (7 methods, 5-fold)

| Experiment | Runs | Time/run | GPU-hours |
|---|---|---|---|
| TOFU 1B (7 × 3 splits × 5 seeds) | 105 | 8 min | 14.0 |
| TOFU 3B (7 × 3 splits × 5 seeds) | 105 | 25 min | 43.8 |
| MUSE Books (7 × 5 seeds) | 35 | 80 min | 46.7 |
| MUSE News (7 × 5 seeds) | 35 | 80 min | 46.7 |
| KnowUnDo (7 × 2 domains × 5 seeds) | 70 | 40 min | 46.7 |
| **Subtotal** | | | **197.9** |

## Finetuning (target models)

| Experiment | Runs | Time/run | GPU-hours |
|---|---|---|---|
| KnowUnDo FT (2 domains) | 2 | 60 min | 2.0 |
| KnowUnDo retrain (2 domains) | 2 | 60 min | 2.0 |
| **Subtotal** | | | **4.0** |

## Ablations (training)

| Experiment | Runs | Time/run | GPU-hours |
|---|---|---|---|
| Inner loop K=0/3/6, 8 ε settings | 24 | 120 min | 48.0 |
| Component ablation (8 configs) | 8 | 104 min | 13.9 |
| Scalability (4 scales × 2 methods) | 8 | 120 min | 16.0 |
| Sustainability (4 steps × 2 methods) | 8 | 120 min | 16.0 |
| **Subtotal** | | | **93.9** |

## Evaluation (generation + metrics)

| Experiment | Runs | Time/run | GPU-hours |
|---|---|---|---|
| TOFU 1B eval | 120 | 5 min | 10.0 |
| TOFU 3B eval | 120 | 8 min | 16.0 |
| MUSE Books + News eval | 80 | 15 min | 20.0 |
| KnowUnDo generation | 84 | 10 min | 14.0 |
| KnowUnDo MMLU (lm_eval) | 20 | 30 min | 10.0 |
| Ablation eval | 48 | 15 min | 12.0 |
| **Subtotal** | | | **82.0** |

## Total

| Category | GPU-hours |
|---|---|
| BLADE training | 40.4 |
| Baseline training | 197.9 |
| Finetuning | 4.0 |
| Ablations | 93.9 |
| Evaluation | 82.0 |
| **Total** | **~450** |
