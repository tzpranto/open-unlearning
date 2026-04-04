# MUSE News (Llama-2-7b-hf)

Generated: 2026-04-04 21:02

## Results

> **Gold (retrain):** 0.3279 | 0.2016 | 0.5602 | 0.0244 | -4.7200  *(forget_knowmem | verbmem | retain | extract | privleak)*


| Method               | forget_knowmem↓ | verbmem↓ | retain↑    | extract↓ | privleak | train_time |
| -------------------- | --------------- | -------- | ---------- | -------- | -------- | ---------- |
| Gold (retrain)       | 0.3279          | 0.2016   | **0.5602** | 0.0244   | -4.7200  | —          |
| Target (pre-unlearn) | 0.6443          | 0.5789   | **0.5552** | 0.2954   | -99.8111 | —          |
| GradAscent           | 0.0000          | 0.0000   | 0.0000     | 0.0079   | 34.9391  | 1h 08m     |
| GradDiff             | 0.0614          | 0.0221   | 0.0427     | 0.0080   | 70.4030  | 1h 34m     |
| NPO                  | 0.6216          | 0.4823   | **0.5336** | 0.2021   | -99.3493 | ⏳          |
| BLURNPO              | ⏳               | ⏳        | ⏳          | ⏳        | ⏳        | ⏳          |
| RMU                  | ⏳               | ⏳        | ⏳          | ⏳        | ⏳        | ⏳          |
| DS-BiAL (ours)       | ⏳               | ⏳        | ⏳          | ⏳        | ⏳        | ⏳          |
| DS-BiAL Exp8r (ref)  | 0.3711          | 0.2110   | 0.4172     | 0.0534   | -99.5592 | 2m 42s     |


## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- `⏳` = run in queue / not yet evaluated
- **bold retain** = within 15% of gold retrain
- `train_time` = wall-clock training only (excl. eval)
- Model: Llama-2-7b-hf, Data: News

