# MUSE News (Llama-2-7b-hf)

Generated: 2026-04-06 (partial — GradAscent/BLURNPO/RMU in queue)

## Results

> **Gold (retrain):** 0.3279 | 0.2016 | 0.5602 | 0.0244 | -4.7200  *(forget_knowmem | verbmem | retain | extract | privleak)*

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | privleak | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3279 | 0.2016 | **0.5602** | 0.0244 | -4.7200 | — |
| Target (pre-unlearn) | 0.6443 | 0.5789 | **0.5552** | 0.2954 | -99.8111 | — |
| GradAscent | 0.0027 | 0.0489 | 0.0077 | 0.0079 | 24.90 | 34m |
| GradDiff | 0.3302 | 0.0053 | 0.2466 | 0.0079 | 108.17 | 1h 2m |
| NPO | 0.5173 | 0.3567 | 0.4195 | 0.0960 | -68.37 | 1h 4m |
| SimNPO | 0.5839 | 0.2585 | **0.4698** | 0.0516 | 71.98 | 58m |
| BLURNPO (β=0.05, LR=2.5e-5) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| RMU (max_steps=80, layers 5-7) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DS-BiAL (ours) | **0.2855** | 0.1479 | 0.2338 | 0.0242 | -98.30 | ~1m |
| DS-BiAL Exp8r (ref) | 0.3711 | 0.2110 | 0.4172 | 0.0534 | -99.56 | ~50s |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold retain** = within 15% of gold retrain
- `train_time` = `train_runtime` from HF `trainer_state.json` (excludes model load/eval) for GradDiff/NPO/SimNPO/GradAscent/BLURNPO/RMU; log-derived (first→last outer step) for DS-BiAL — not directly comparable
- Model: Llama-2-7b-hf, Data: News
