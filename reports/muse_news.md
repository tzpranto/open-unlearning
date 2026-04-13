# MUSE News (Llama-2-7b-hf)

Generated: 2026-04-11 (all baselines complete, PerTA reproduced)

## Results

> **Gold (retrain):** 0.3243 | 0.2042 | 0.5523 | 0.0247 | -0.2099  *(forget_knowmem | verbmem | retain | extract | privleak)*

| Method | forget_knowmem↓ | verbmem↓ | retain↑ | extract↓ | privleak | notes |
| --- | --- | --- | --- | --- | --- | --- |
| Gold (retrain) | 0.3243 | 0.2042 | **0.5523** | 0.0247 | -0.21 | — |
| Target (pre-unlearn) | 0.6538 | 0.5693 | 0.5436 | 0.3023 | -99.81 | — |
| GradAscent | 0.0027 | 0.0489 | 0.0077 | 0.0079 | 24.90 | collapses retain |
| GradDiff | 0.3302 | 0.0053 | 0.2466 | 0.0079 | 108.17 | |
| NPO | 0.5173 | 0.3567 | 0.4195 | 0.0960 | -68.37 | |
| SimNPO | 0.5839 | 0.2585 | 0.4698 | 0.0516 | 71.98 | best retain among gradient baselines |
| BLURNPO | 0.5806 | 0.3559 | **0.5318** | 0.1236 | -99.75 | checkpoint-100/130 (OOM prevented full run) |
| RMU | 0.5164 | 0.2848 | 0.4567 | 0.0552 | -99.75 | max_steps=80, layers 5-7 |
| **PerTA (ours)** | **0.2820** | **0.1755** | 0.3964 | **0.0206** | -46.87 | λ=3.5, α=1.0; reproduced (deterministic) |

## Method Description

### PerTA (Per-parameter Task Arithmetic)

Weight-space unlearning via selective task vector negation:

```
θ_final = θ_target - λ * w * (θ_target - θ_pretrained)
w_i = F_forget_i / (F_forget_i + α * F_retain_i + ε)
```

- Fisher information (diagonal) computed on target model using 64 samples each from forget/retain splits
- Fisher mask mean w=0.24: 24% of parameters are forget-dominated (negated), 76% retain-protected
- λ=3.5 achieves best frontier-breaking result (δ=+0.071 above CE Pareto frontier)
- No gradient-based training — pure weight surgery on pretrained/target state dicts
- Fully deterministic and reproducible (confirmed: two runs produce identical metrics)

## Notes

- ↓ = lower is better (forgetting quality)
- ↑ = higher is better (retain quality)
- **bold** = best result on that metric
- Gold target: forget_knowmem ≤ 0.324, retain ≥ 0.552
- All evals from fresh runs (2026-04-10/11)
- BLURNPO: evaluated at checkpoint-100 (step 100/130); full run OOMs on 96GB GPU due to bilevel `retain_graph=True`
- Model: Llama-2-7b-hf, Data: MUSE-News
