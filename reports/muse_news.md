# MUSE News (Llama-2-7b-hf)

Generated: 2026-04-05 (partial — NPO/SimNPO/DS-BiAL/BLURNPO-v2/RMU-v2 in queue)

## Results

> **Gold (retrain):** 0.3279 | 0.2016 | 0.5602 | 0.0244 | -4.7200  *(forget_knowmem | verbmem | retain | extract | privleak)*

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | privleak | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3279 | 0.2016 | **0.5602** | 0.0244 | -4.7200 | — |
| Target (pre-unlearn) | 0.6443 | 0.5789 | **0.5552** | 0.2954 | -99.8111 | — |
| GradAscent | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| GradDiff | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| NPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SimNPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| BLURNPO (β=0.05, LR=2.5e-5) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| RMU (max_steps=80, layers 5-7) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DS-BiAL (ours) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DS-BiAL Exp8r (ref) | 0.3711 | 0.2110 | 0.4172 | 0.0534 | -99.5592 | 2m 42s |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold retain** = within 15% of gold retrain
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: News
