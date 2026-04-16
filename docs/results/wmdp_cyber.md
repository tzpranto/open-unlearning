# WMDP Cyber (Zephyr-7b-beta)

Updated: 2026-04-15

Target: wmdp_cyber accuracy ↓ (random=25%), MMLU ↑ (preserve general knowledge)

## Baselines

| Method | wmdp_cyber↓ | mmlu↑ | train_time | notes |
| --- | --- | --- | --- | --- |
| Zephyr (no unlearn) | **0.431** | **0.589** | — | pre-unlearning baseline |
| Random guess | 0.250 | 0.250 | — | lower bound |

## Our Experiments

| Method | wmdp_cyber↓ | mmlu↑ | train_time | notes |
| --- | --- | --- | --- | --- |
| LoRA-Imp LM (K=3, 2ep) | — | — | — | wmdp_cy_imp_lm |
| LoRA-Imp K=5 (lr=1e-4) | — | — | — | wmdp_cy_imp_k5 |
| GSP-SIBL base (a=0,b=0) | — | — | — | wmdp_cy_gsp_base |
| GSP-SIBL a=5,b=5 | — | — | — | wmdp_cy_gsp_a5b5 |
| GSP-SIBL a=20 K=5 | — | — | — | wmdp_cy_gsp_a20_k5 |
| GSP-SIBL a=5 +recovery | — | — | — | wmdp_cy_gsp_rec |

## Data

- Forget: 1,000 samples — exploit code, WMI evasion, backdoor stagers, command injection
- Retain: 4,473 samples — benign code (requirements.txt, SoftFloat docs, OpenBSD libm)
- Domains are **semantically orthogonal** — GSP should find real subspace structure
- No bio-forget-corpus available locally (bio track deferred)

## Why WMDP after MUSE News

MUSE News GSP was mathematically futile: E_f std=0.009, weight spread capped at 1.9%.
Both forget/retain are BBC News articles — same domain, same gradient subspace.

WMDP Cyber has fundamentally different domains (exploit code vs benign code).
Expected E_f std ~0.05-0.15 giving meaningful GSP weight differentiation.

## Key Findings (will update)

