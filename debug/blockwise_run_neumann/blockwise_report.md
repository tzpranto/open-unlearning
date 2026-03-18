# Blockwise Implicit Experiment

## Config
- solver: `neumann`
- neumann_variant: `richardson`
- last_n_layers: `2`
- include_attn: `True`
- include_mlp: `True`
- T: `2`
- K: `1`
- neumann_mu: `1.0`
- neumann_steps: `2`
- neumann_alpha_default: `0.01`
- neumann_use_probe_alpha: `False`
- cg_damping: `0.1`
- cg_iters: `20`

## Selected Blocks
- layer_30: params=9, dim=202383360
- layer_31: params=9, dim=202383360

## Iteration Summary

| iter | L_fgt | L_ret | r | lambda | sec |
|---:|---:|---:|---:|---:|---:|
| 0 | 24.3059 | 0.4795 | +0.3795 | 0.3795 | 1.46 |
| 1 | 23.4261 | 0.3617 | +0.2617 | 0.6412 | 1.14 |

- JSON report: `debug/blockwise_run_neumann/blockwise_report.json`
