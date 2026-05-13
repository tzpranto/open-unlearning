#!/usr/bin/env python3
"""Generate K=0 vs K=3 vs K=6 inner loop ablation figure from ablation.md data."""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

with open("results/ablation.md") as f:
    content = f.read()

# Extract the 3-way comparison table (K=0, K=3, K=6)
table_match = re.search(
    r"\| eps_mul \| K=0 HM \| K=3 HM \| K=6 HM \| Best K \| K=0 retain \| K=3 retain \| K=6 retain \|\n((?:\|.*\n)+)",
    content
)
rows = table_match.group(1).strip().split("\n")

eps_muls, k0_hm, k3_hm, k6_hm, k0_ret, k3_ret, k6_ret = [], [], [], [], [], [], []
for row in rows:
    cols = [c.strip().replace("**", "") for c in row.split("|")[1:-1]]
    if cols[0].startswith("---") or cols[0] == "":
        continue
    eps_muls.append(float(cols[0]))
    k0_hm.append(float(cols[1]))
    k3_hm.append(float(cols[2]))
    k6_hm.append(float(cols[3]))
    k0_ret.append(float(cols[5]))
    k3_ret.append(float(cols[6]))
    k6_ret.append(float(cols[7]))

eps_muls = np.array(eps_muls)
k0_hm = np.array(k0_hm)
k3_hm = np.array(k3_hm)
k6_hm = np.array(k6_hm)
k0_ret = np.array(k0_ret)
k3_ret = np.array(k3_ret)
k6_ret = np.array(k6_ret)

fig, ax1 = plt.subplots(1, 1, figsize=(4, 2.8))
plt.style.use('seaborn-v0_8-whitegrid')

x = np.arange(len(eps_muls))
xlabels = [str(e) for e in eps_muls]

ax1.plot(x, k3_hm, 'o-', color='#2166ac', linewidth=1.8, markersize=5, label='$K{=}3$ (ours)')
ax1.plot(x, k6_hm, '^-', color='#4daf4a', linewidth=1.8, markersize=5, label='$K{=}6$')
ax1.plot(x, k0_hm, 's--', color='#b2182b', linewidth=1.8, markersize=5, label='$K{=}0$ (no inner loop)')
ax1.axhline(k3_hm.mean(), color='#2166ac', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.axhline(k6_hm.mean(), color='#4daf4a', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.axhline(k0_hm.mean(), color='#b2182b', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.text(7.1, k3_hm.mean() + 0.002, f'avg={k3_hm.mean():.3f}', fontsize=7, color='#2166ac', va='bottom')
ax1.text(7.1, k6_hm.mean() - 0.002, f'avg={k6_hm.mean():.3f}', fontsize=7, color='#4daf4a', va='top')
ax1.text(7.1, k0_hm.mean() - 0.002, f'avg={k0_hm.mean():.3f}', fontsize=7, color='#b2182b', va='top')
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels, fontsize=9)
ax1.set_xlabel(r'Constraint tightness ($\epsilon$ multiplier) $\longrightarrow$ looser', fontsize=9.5)
ax1.set_ylabel('Composite Score (HM $\\uparrow$)', fontsize=9.5)
ax1.legend(fontsize=8.5, loc='lower right')
ax1.set_ylim(0.44, 0.57)

plt.tight_layout()
plt.savefig('paper/figures/fig_inner_loop_ablation.pdf', bbox_inches='tight', dpi=600)
plt.savefig('docs/design/figures/fig_inner_loop_ablation.pdf', bbox_inches='tight', dpi=600)
plt.savefig('docs/design/figures/fig_inner_loop_ablation.png', bbox_inches='tight', dpi=200)
print(f"K=0 avg HM: {k0_hm.mean():.3f}, K=3 avg HM: {k3_hm.mean():.3f}, K=6 avg HM: {k6_hm.mean():.3f}")
print(f"K=0->K=3: +{(k3_hm.mean()-k0_hm.mean()):.3f}, K=3->K=6: {(k6_hm.mean()-k3_hm.mean()):+.3f}")
print("Saved fig_inner_loop_ablation.pdf and .png")
