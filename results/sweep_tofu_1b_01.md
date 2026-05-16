# BLADE Hyperparameter Sweep — TOFU 1B forget01 (seed=42)

Updated: 2026-05-15

## Setup

- **Model**: Llama-3.2-1B-Instruct (open-unlearning/tofu_Llama-3.2-1B-Instruct_full)
- **Split**: forget01 / retain99 / holdout01
- **Seed**: 42
- **Base config**: `experiment=unlearn/tofu/lora_bial_1b.yaml`
- **Baseline params**: eta_theta=5e-5, eta_in=2e-4, eps_mul=0.85, tau=0.7, rho=0.1, alpha_dual=0.1, T=250, K=3, lora_r=8
- **GPUs**: 8x A100-40GB (parallel batches of 8)
- **Baseline result (paper)**: MU=0.599, Prob=0.002, ROUGE=0.038, HM=0.808

---

## K=3 Results

### Sweep 1: eps_mul

| eps_mul | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.75 | 0.5994 | 0.0016 | 0.0335 | 0.8098 | 113 |
| 1.0 | 0.5997 | 0.0017 | 0.0345 | 0.8097 | 113 |
| 1.25 | 0.5982 | 0.0013 | 0.0303 | 0.8099 | 250 |
| 1.5 | 0.5969 | 0.0011 | 0.0313 | 0.8089 | 250 |
| 2.0 | 0.5951 | 0.0010 | 0.0328 | 0.8075 | 250 |
| 2.5 | 0.6015 | 0.0017 | 0.0323 | 0.8114 | 250 |
| 3.0 | 0.5969 | 0.0000 | 0.0129 | **0.8133** | 250 |
| 3.2 | 0.5957 | 0.0001 | 0.0207 | 0.8108 | 250 |

**Range**: 0.8075–0.8133 (Δ=0.006). Insensitive. Looser constraints give slightly better forgetting (lower ROUGE/Prob) without hurting retain.

### Sweep 2: tau (clamped_entropy_tau)

| tau | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.1 | 0.5984 | 0.0156 | 0.1790 | 0.7682 | 250 |
| 0.2 | 0.6025 | 0.0048 | 0.0555 | 0.8057 | 250 |
| 0.3 | 0.6054 | 0.0009 | 0.0437 | 0.8112 | 131 |
| 0.4 | 0.6066 | 0.0007 | 0.0458 | 0.8114 | 115 |
| 0.5 | 0.6039 | 0.0007 | 0.0362 | 0.8121 | 106 |
| 0.6 | 0.6052 | 0.0016 | 0.0386 | 0.8121 | 106 |
| 0.9 | 0.6033 | 0.0007 | 0.0310 | **0.8130** | 127 |
| 1.0 | 0.5933 | 0.0000 | 0.0293 | 0.8073 | 166 |

**Range**: 0.7682–0.8130 (Δ=0.045). **Only sensitive param**. tau<0.2 clearly degrades. Sweet spot: 0.5–0.9. Baseline tau=0.7 is in the optimal region.

### Sweep 3: alpha_dual (dual_decay_factor)

| alpha_dual | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.01 | 0.5977 | 0.0017 | 0.0378 | 0.8078 | 119 |
| 0.05 | 0.5992 | 0.0013 | 0.0376 | 0.8088 | 110 |
| 0.1 | 0.5997 | 0.0019 | 0.0418 | 0.8080 | 113 |
| 0.2 | 0.5985 | 0.0014 | 0.0407 | 0.8076 | 110 |
| 0.3 | 0.5963 | 0.0013 | 0.0389 | 0.8067 | 113 |
| 0.5 | 0.5982 | 0.0019 | 0.0366 | 0.8083 | 119 |
| 0.7 | 0.5973 | 0.0015 | 0.0338 | 0.8085 | 113 |
| 1.0 | 0.5992 | 0.0018 | 0.0366 | **0.8089** | 119 |

**Range**: 0.8067–0.8089 (Δ=0.002). **Completely insensitive**. Symmetric vs asymmetric makes no difference on TOFU 1B.

### Sweep 4: rho

| rho | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.01 | 0.5920 | 0.0015 | 0.0351 | 0.8049 | 106 |
| 0.03 | 0.5999 | 0.0016 | 0.0387 | **0.8089** | 112 |
| 0.05 | 0.5993 | 0.0018 | 0.0401 | 0.8081 | 113 |
| 0.1 | 0.5997 | 0.0019 | 0.0418 | 0.8080 | 113 |
| 0.2 | 0.5952 | 0.0013 | 0.0403 | 0.8057 | 119 |
| 0.5 | 0.5972 | 0.0017 | 0.0459 | 0.8055 | 136 |
| 1.0 | 0.5957 | 0.0015 | 0.0373 | 0.8066 | 250 |
| 2.0 | 0.5966 | 0.0020 | 0.0467 | 0.8049 | 250 |

**Range**: 0.8049–0.8089 (Δ=0.004). Mostly insensitive. Very high rho (≥1.0) prevents early convergence. Sweet spot: 0.03–0.1.

### Sweep 5: eta_in (eta_theta fixed at 5e-5)

| eta_in | ratio | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|---|
| 1e-5 | 0.2x | 0.5970 | 0.0015 | 0.0369 | 0.8076 | 113 |
| 2.5e-5 | 0.5x | 0.5997 | 0.0014 | 0.0338 | 0.8100 | 112 |
| 5e-5 | 1x | 0.6047 | 0.0013 | 0.0372 | **0.8122** | 113 |
| 1e-4 | 2x | 0.5986 | 0.0014 | 0.0354 | 0.8089 | 113 |
| 2e-4 | 4x | 0.5997 | 0.0019 | 0.0418 | 0.8080 | 113 |
| 5e-4 | 10x | 0.5931 | 0.0020 | 0.0388 | 0.8046 | 113 |
| 1e-3 | 20x | 0.5706 | 0.0018 | 0.0390 | 0.7905 | 201 |
| 2e-3 | 40x | 0.5777 | 0.0004 | 0.0317 | 0.7970 | 217 |

**Range**: 0.7905–0.8122 (Δ=0.022). Moderate sensitivity. eta_in=5e-5 (ratio 1×) is best. High inner LR (≥1e-3) causes retain oscillation. Baseline 2e-4 (4×) is adequate but not optimal.

---

## K=0 Results

### Sweep 1: eps_mul

| eps_mul | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.75 | 0.6010 | 0.0013 | 0.0368 | 0.8101 | 112 |
| 1.0 | 0.6008 | 0.0013 | 0.0319 | 0.8111 | 112 |
| 1.25 | 0.6030 | 0.0014 | 0.0322 | 0.8123 | 112 |
| 1.5 | 0.5992 | 0.0014 | 0.0330 | 0.8099 | 111 |
| 2.0 | 0.6011 | 0.0014 | 0.0319 | 0.8112 | 105 |
| 2.5 | 0.5993 | 0.0001 | 0.0197 | **0.8133** | 250 |
| 3.0 | 0.5997 | 0.0001 | 0.0227 | 0.8128 | 250 |
| 3.2 | 0.5988 | 0.0001 | 0.0211 | 0.8126 | 250 |

**Range**: 0.8099–0.8133 (Δ=0.003). Insensitive.

### Sweep 2: tau (clamped_entropy_tau)

| tau | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.1 | 0.6015 | 0.0146 | 0.1689 | 0.7731 | 250 |
| 0.2 | 0.6054 | 0.0043 | 0.0405 | 0.8111 | 250 |
| 0.3 | 0.6102 | 0.0010 | 0.0443 | 0.8139 | 123 |
| 0.4 | 0.6131 | 0.0006 | 0.0456 | 0.8154 | 112 |
| 0.5 | 0.6159 | 0.0005 | 0.0428 | **0.8177** | 105 |
| 0.6 | 0.6024 | 0.0016 | 0.0450 | 0.8089 | 105 |
| 0.9 | 0.6035 | 0.0005 | 0.0338 | 0.8124 | 120 |
| 1.0 | 0.6008 | 0.0013 | 0.0319 | 0.8111 | 112 |

**Range**: 0.7731–0.8177 (Δ=0.045). **Only sensitive param**. Same pattern as K=3: tau<0.2 breaks forgetting. Sweet spot shifts slightly lower (0.4–0.5 vs 0.5–0.9).

### Sweep 3: alpha_dual (dual_decay_factor)

| alpha_dual | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.01 | 0.5967 | 0.0014 | 0.0342 | 0.8080 | 111 |
| 0.05 | 0.6037 | 0.0014 | 0.0305 | **0.8131** | 106 |
| 0.1 | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 0.2 | 0.6006 | 0.0013 | 0.0291 | 0.8116 | 111 |
| 0.3 | 0.5959 | 0.0012 | 0.0371 | 0.8069 | 112 |
| 0.5 | 0.6029 | 0.0014 | 0.0318 | 0.8124 | 112 |
| 0.7 | 0.5979 | 0.0013 | 0.0300 | 0.8098 | 112 |
| 1.0 | 0.5988 | 0.0012 | 0.0299 | 0.8103 | 111 |

**Range**: 0.8069–0.8131 (Δ=0.006). Insensitive.

### Sweep 4: rho

| rho | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|
| 0.01 | 0.5972 | 0.0014 | 0.0225 | 0.8110 | 102 |
| 0.03 | 0.6023 | 0.0014 | 0.0299 | 0.8124 | 106 |
| 0.05 | 0.6056 | 0.0016 | 0.0350 | **0.8132** | 105 |
| 0.1 | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 0.2 | 0.6063 | 0.0011 | 0.0378 | 0.8131 | 112 |
| 0.5 | 0.6038 | 0.0013 | 0.0413 | 0.8107 | 120 |
| 1.0 | 0.6017 | 0.0015 | 0.0454 | 0.8084 | 129 |
| 2.0 | 0.6002 | 0.0012 | 0.0449 | 0.8077 | 130 |

**Range**: 0.8077–0.8132 (Δ=0.006). Mostly insensitive. Same pattern: high rho delays convergence.

### Sweep 5: eta_in (eta_theta fixed at 5e-5)

| eta_in | ratio | MU | Prob↓ | ROUGE↓ | HM↑ | stop_step |
|---|---|---|---|---|---|---|
| 1e-5 | 0.2x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 2.5e-5 | 0.5x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 5e-5 | 1x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 1e-4 | 2x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 2e-4 | 4x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 5e-4 | 10x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 1e-3 | 20x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |
| 2e-3 | 40x | 0.5982 | 0.0014 | 0.0313 | 0.8097 | 112 |

**All identical**. eta_in is provably irrelevant at K=0 (no inner loop executed).

---

## Summary: K=3 vs K=0

| Param | K=3 Best HM | K=3 Δ | K=0 Best HM | K=0 Δ | Winner |
|---|---|---|---|---|---|
| eps_mul | 0.8133 | 0.006 | 0.8133 | 0.003 | Tie |
| **tau** | 0.8130 | **0.045** | **0.8177** | **0.045** | **K=0** |
| alpha_dual | 0.8089 | 0.002 | 0.8131 | 0.006 | K=0 |
| rho | 0.8089 | 0.004 | 0.8132 | 0.006 | K=0 |
| eta_in | 0.8122 | 0.022 | 0.8097 | 0.000 | K=3 |

**Key findings:**
1. K=0 matches or slightly beats K=3 on TOFU 1B forget01 — inner retain-repair loop adds no value here (low entanglement)
2. tau is the only sensitive param in both settings (tau < 0.2 breaks forgetting)
3. eta_in is provably irrelevant at K=0 (all 8 values identical); at K=3 only extreme values (≥20×) hurt
4. Overall best across all 80 runs: **K=0, tau=0.5 → HM=0.8177**

---

## Lambda trajectory analysis (pending)

Each run saves step-by-step training dynamics in `lora_bial_history.json`:
- Fields: step, L_fgt, L_ret, r (residual), lambda, inner_loss_mean, vel_ema, dt, lr
- **K=3 logs**: `saves/unlearn/sweep_{param}_{value}/lora_bial_history.json` (40 files)
- **K=0 logs**: `saves/unlearn/k0_sweep_{param}_{value}/lora_bial_history.json` (40 files)

Key questions to answer from these logs:
- How does λ grow/stabilize under different rho and eps_mul?
- What happens to λ when eta_in is too high (1e-3, 2e-3) — oscillation/failure mode?
- How does K=0 compensate via λ for lacking the inner loop?
- Does λ clip/floor matter? (lambda_min=0.1 is the only projection applied)

## TODO

- [x] K=3 sweep: 40 runs complete (2026-05-15)
- [x] K=0 sweep: 40 runs complete (2026-05-15)
- [ ] Analyse lambda trajectories from all 80 runs (K=3 + K=0)
- [ ] K=0 baseline on TOFU 1B forget05 and forget10 (default params, seed=42)
- [ ] MUSE sensitivity sweep (tau + eps_mul at minimum)
- [ ] Multi-seed λ trajectories — check if existing 5-seed paper runs have saved histories
