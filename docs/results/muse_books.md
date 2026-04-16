# MUSE Books (Llama-2-7b-hf)

Generated: 2026-04-04 21:02

## Results

> **Gold (retrain):** 0.3029 | 0.1445 | 0.6874 | 0.0107 | 8.1600  *(forget_knowmem | verbmem | retain | extract | privleak)*

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | privleak | train_time |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3029 | 0.1445 | **0.6874** | 0.0107 | 8.1600 | — |
| Target (pre-unlearn) | 0.4712 | 0.9970 | **0.6913** | 0.9163 | -57.3410 | — |
| GradAscent | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| GradDiff | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| NPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| BLURNPO | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| RMU | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| DS-BiAL (ours) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold retain** = within 15% of gold retrain
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: Books
