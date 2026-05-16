# Lambda Trajectory Analysis: BLADE Hyperparameter Sweep (TOFU 1B, forget01)

## Overview

Analyzed 80 runs (40 K=3, 40 K=0) across 5 hyperparameters with 8 values each.
All 80 expected files were present and successfully loaded.

**Lambda update rule recap:** lambda_{t+1} = lambda_t + rho * r_t, where r = (L_ret - eps) / eps.
When r > 0 (constraint violated), lambda increases; when r < 0 (constraint satisfied), lambda decreases
(scaled by alpha_dual for asymmetric decay).

---

## 1. Effect of rho (Quadratic Penalty / Dual Step Size)

**Key finding:** rho is the single strongest determinant of final lambda magnitude. It controls
both the step size of dual updates and the quadratic penalty weight.

### K=3 Trajectories (lambda at steps 0, 25, 50, 75, 100, final):
```
rho=0.01 : s0=1.001  s25=1.005  s50=1.161  s75=1.239  s100=1.256  final=1.258   (conv step 105)
rho=0.03 : s0=1.002  s25=1.016  s50=1.414  s75=1.603  s100=1.653  final=1.664   (conv step 111)
rho=0.05 : s0=1.003  s25=1.027  s50=1.595  s75=1.867  s100=1.946  final=1.966   (conv step 112)
rho=0.1  : s0=1.006  s25=1.051  s50=1.916  s75=2.430  s100=2.586  final=2.631   (conv step 112)
rho=0.2  : s0=1.013  s25=1.097  s50=2.410  s75=3.168  s100=3.434  final=3.558   (conv step 118)
rho=0.5  : s0=1.032  s25=1.244  s50=3.363  s75=4.786  s100=5.435  final=5.938   (conv step 135)
rho=1.0  : s0=1.065  s25=1.487  s50=4.460  s75=6.616  s100=7.978  final=10.841  (NO convergence)
rho=2.0  : s0=1.130  s25=1.963  s50=6.258  s75=8.702  s100=10.962 final=16.376  (NO convergence)
```

**Observations:**
- Final lambda scales roughly linearly with rho for small values (0.01-0.2).
- At rho >= 1.0, lambda grows without bound (no convergence in 250 steps) -- the dual variable
  overshoots, creating oscillation that prevents the constraint from being stably satisfied.
- K=3 mean final lambda for rho sweep: 5.53 (std=5.40) -- high variance driven by large-rho runs.
- **Sweet spot:** rho in [0.05, 0.2] yields convergent runs with moderate lambda (1.9-3.6).

### K=0 vs K=3 at rho=1.0:
```
K=3: final=10.841 (250 steps, no convergence)
K=0: final=5.760  (129 steps, converged)
```
K=0 converges even at rho=1.0 because the outer-only update is less aggressive, requiring
less lambda pressure. K=3's inner loop drives stronger forgetting, pushing L_ret higher and
demanding more lambda compensation.

---

## 2. Effect of eps_mul (Constraint Tightness)

**Key finding:** eps_mul directly controls whether the dual variable rises or falls in late training.
Tight constraints (small eps_mul) force lambda high; loose constraints (large eps_mul)
cause lambda to decay after a mid-training peak.

### K=3 Results:
```
eps_mul  init    final   max     conv?   r_final
0.75     1.008   2.733   2.733   step112  0.0179   (tight: lambda monotonically rises to high value)
1.0      1.004   2.485   2.485   step112  0.0355
1.25     1.001   2.223   2.224   step110 -0.0546
1.5      1.000   2.092   2.104   step114 -0.0416
2.0      0.999   1.770   1.804   step108 -0.1152
2.5      0.998   1.721   1.787   step108 -0.1763
3.0      0.998   1.075   1.547   NONE    -0.2750   (loose: lambda peaks at step 60 then DECAYS)
3.2      0.997   1.088   1.610   NONE    -0.2830
```

**Critical behavior at large eps_mul (3.0, 3.2):**
- Lambda peaks mid-training (step ~60, reaching ~1.55) then steadily declines.
- At peak, r briefly becomes positive (L_ret exceeds eps), causing lambda to rise.
- After forgetting succeeds, L_ret stays well below the loose threshold (r ~ -0.28),
  so lambda decays at alpha_dual * rho * |r| per step.
- Final lambda (1.08) is barely above 1.0, meaning almost no retention pressure remains.
- This explains why large eps_mul leads to high Prob/ROUGE (over-forgetting with no check).

**eps_mul=0.75 (tight):** lambda rises monotonically to 2.73 because L_ret persistently
bumps against the tight constraint. Final r=0.018 (barely above zero) indicates the
system found equilibrium with strong retention pressure.

### Early-step dip for loose eps_mul:
```
eps_mul=0.75: 1.008 -> 1.012 -> 1.016 -> ... (always rising)
eps_mul=2.5:  0.998 -> 0.996 -> 0.995 -> ... (decreasing for first ~30 steps)
eps_mul=3.2:  0.998 -> 0.995 -> 0.992 -> ... (decreasing for first ~30 steps)
```
With large eps_mul, the initial eps threshold is so generous that r < 0 from the start,
causing lambda to decrease before forgetting dynamics push L_ret up.

---

## 3. Effect of High eta_in (Inner Learning Rate) at K=3

**Key finding:** High eta_in (1e-3, 2e-3) causes lambda to grow much higher (4.1) than
moderate eta_in values (2.5-2.7), and requires nearly 2x more steps to converge.

### Comparison:
```
eta_in    final_lambda  conv_step  total_steps  L_ret_final
5e-5      2.602         112        113          0.0920     (well-behaved)
1e-4      2.610         112        113          0.2015
5e-4      2.513         112        113          0.1408
1e-3      4.114         200        201          0.1762     (degraded)
2e-3      4.075         216        217          0.3009     (degraded, high L_ret)
```

**Mechanism:** With high inner LR, the inner loop takes excessively large steps that
momentarily spike L_ret (retention loss), causing r > 0 and driving lambda up. The
outer loop then overcorrects. This tug-of-war extends training significantly.

**Trajectory for eta_in=2e-3:**
```
step   0: lambda=1.006  L_ret=0.180
step  50: lambda=1.342  L_ret=0.219
step 100: lambda=1.976  L_ret=0.170
step 150: lambda=2.835  L_ret=0.143
step 200: lambda=3.801  L_ret=0.423   (L_ret spike!)
FINAL   : lambda=4.075  L_ret=0.301
```
The intermittent L_ret spikes (e.g., 0.42 at step 200) keep driving lambda upward.

**K=0 is completely insensitive to eta_in** (all 8 runs produce identical lambda=2.338)
because K=0 has no inner loop -- eta_in is unused.

---

## 4. K=0 vs K=3: How Does K=0 Compensate?

### Overall pattern:
- K=0 consistently produces **lower final lambda** than K=3 (ratio 0.48-0.97 depending on setting).
- K=0 converges faster (fewer steps) in most cases.

### Detailed comparison (rho sweep):
```
rho    K3_final  K0_final  K0/K3_ratio
0.01    1.258     1.220     0.97
0.1     2.631     2.338     0.89
0.5     5.938     4.587     0.77
1.0    10.841     5.760     0.53
2.0    16.376     7.863     0.48
```

**Interpretation:** K=0 (outer-only updates) produces smaller L_ret perturbations per step,
so the constraint is less frequently violated, and lambda accumulates less pressure.
At large rho, this difference is amplified: K=3's aggressive inner updates keep pushing
L_ret above eps, while K=0's gentler single-step updates stay closer to the constraint boundary.

The tradeoff: K=0 needs less lambda because it's less effective at forgetting.
K=3 drives L_fgt down faster but at the cost of more retention damage, which the
dual variable must counteract.

---

## 5. Effect of alpha_dual (Asymmetric Decay Factor)

**Key finding:** alpha_dual has minimal effect on the lambda trajectory in these runs.

### K=3 Results:
```
alpha_dual  final_lambda  std_over_values
0.01        2.668         
0.05        2.536         
0.1         2.631         (default)
0.2         2.552         
0.3         2.615         
0.5         2.644         
0.7         2.632         
1.0         2.598         

Mean: 2.610, Std: 0.046   <-- extremely low variance!
```

**Why so little effect?** In these successful runs (rho=0.1, moderate eps_mul=1.5), the
constraint residual r stays positive for most of training (L_ret > eps during the active
forgetting phase). alpha_dual only scales the *negative* updates (when r < 0), which
happen infrequently and briefly. The dominant regime is r > 0 (lambda increasing), which
is unaffected by alpha_dual.

**alpha_dual would matter more** in settings where the constraint oscillates around
satisfaction (e.g., late training or loose eps_mul). In the eps_mul=3.0 case, lambda
decays from 1.55 to 1.08 -- here alpha_dual would control that decay rate.

### Growth rate comparison (steps 25-75):
```
alpha_dual=0.01: avg delta_lambda/step = 0.02804 (0 negative steps out of 50)
alpha_dual=0.1:  avg delta_lambda/step = 0.02758 (0 negative steps out of 50)
alpha_dual=0.5:  avg delta_lambda/step = 0.02786 (0 negative steps out of 50)
alpha_dual=1.0:  avg delta_lambda/step = 0.02685 (0 negative steps out of 50)
```
Zero negative steps in the 25-75 range confirms alpha_dual is never activated during
the main training phase.

---

## 6. Effect of tau (Clamped Entropy Threshold)

**Key finding:** Low tau suppresses lambda growth; high tau amplifies it. tau=0.1 produces
the lowest final lambda across all sweeps (1.56 at K=3).

### K=3 Results:
```
tau    final_lambda  conv_step  r_final  
0.1    1.563         NONE       -0.0032   (never converges, lambda stays low)
0.2    1.940         NONE       -0.0128   (never converges)
0.3    1.984         130         0.0131
0.4    2.087         114         0.0175
0.5    2.166         105         0.0301   
0.6    2.176         105         0.0546
0.9    3.159         126         0.0180
1.0    3.455         165        -0.0040
```

**Mechanism:** tau controls the clamped entropy term in the forget loss. Low tau clamps
early, making L_fgt small from the start:
```
tau=0.1: L_fgt starts at 0.685 (already low due to clamping)
tau=1.0: L_fgt starts at 7.17 (full entropy)
```

When L_fgt is small, the outer update barely perturbs retention (small gradient signal),
so L_ret stays below eps, r stays near zero or negative, and lambda barely grows.
This is why low-tau runs produce high Prob/ROUGE in downstream metrics -- the dual
variable never builds enough pressure to protect retention, because the forgetting
signal is too weak to *need* protection.

**tau=0.9, 1.0 (high):** Lambda rises to 3.2-3.5 because strong forgetting gradients
push L_ret well above eps for an extended period.

### tau=0.1 detailed trajectory:
```
step   0: lambda=1.006  L_fgt=0.685  L_ret=0.180
step  50: lambda=1.166  L_fgt=0.324  L_ret=0.145
step 100: lambda=1.369  L_fgt=0.060  L_ret=0.156
step 200: lambda=1.520  L_fgt=0.013  L_ret=0.108
FINAL   : lambda=1.563  L_fgt=0.016  L_ret=0.112
```
Lambda grows only 0.56 total over 250 steps -- the dual variable barely engages.

---

## Summary of Key Findings

| Parameter | Effect on Lambda | Mechanism |
|-----------|-----------------|-----------|
| **rho** | Strong positive (final lambda scales ~linearly with rho) | Directly multiplies dual step size; high rho causes overshoot/non-convergence |
| **eps_mul** | Strong negative (tight constraint => high lambda) | Tight eps keeps r>0 longer; loose eps causes lambda decay after peak |
| **eta_in** | Moderate (high LR => ~60% higher lambda at K=3) | Aggressive inner steps spike L_ret, triggering persistent constraint violation |
| **tau** | Strong positive (high tau => high lambda) | High tau = stronger forgetting signal = more L_ret perturbation |
| **alpha_dual** | Negligible (<2% std across values) | Only affects negative updates, which rarely occur in converging runs |
| **K=0 vs K=3** | K=0 produces 10-50% lower lambda | K=0's gentler updates cause fewer constraint violations |

### Practical Implications for Tuning:
1. **rho is the primary knob** for lambda dynamics. Keep rho in [0.05, 0.2] for stable convergence.
2. **eps_mul > 2.5** causes lambda to decay in late training -- the system "gives up" on retention.
3. **eta_in > 5e-4 at K=3** causes instability in the dual variable; stick to 5e-5-2e-4.
4. **tau < 0.3** starves the dual mechanism -- lambda never rises enough to protect retention.
5. **alpha_dual** can be safely set anywhere in [0.01, 1.0] without affecting behavior in normal operating regimes.
