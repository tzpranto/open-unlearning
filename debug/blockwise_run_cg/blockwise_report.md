# Blockwise Implicit Experiment

## Config
- solver: `cg`
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
| 0 | 24.2998 | 0.4797 | +0.3797 | 0.3797 | 24.30 |
| 1 | 8.7106 | 8.0009 | +7.9009 | 8.2806 | 23.82 |

- JSON report: `debug/blockwise_run_cg/blockwise_report.json`
