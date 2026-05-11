#!/usr/bin/env python3
"""Generate K=0 vs K=3 inner loop ablation figure from ablation.md data."""
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Parse data directly from ablation.md
with open("results/ablation.md") as f:
    content = f.read()

# Extract K=0 vs K=3 comparison table
table_match = re.search(
    r"\| eps_mul \| K=3 HM \| K=0 HM \| Δ HM \| K=3 retain \| K=0 retain \| Δ retain \|\n((?:\|.*\n)+)",
    content
)
rows = table_match.group(1).strip().split("\n")

eps_muls, k3_hm, k0_hm, k3_ret, k0_ret = [], [], [], [], []
for row in rows:
    cols = [c.strip() for c in row.split("|")[1:-1]]
    if cols[0].startswith("---") or cols[0] == "":
        continue
    eps_muls.append(float(cols[0]))
    k3_hm.append(float(cols[1]))
    k0_hm.append(float(cols[2]))
    k3_ret.append(float(cols[4]))
    k0_ret.append(float(cols[5]))

eps_muls = np.array(eps_muls)
k3_hm = np.array(k3_hm)
k0_hm = np.array(k0_hm)
k3_ret = np.array(k3_ret)
k0_ret = np.array(k0_ret)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
plt.style.use('seaborn-v0_8-whitegrid')

x = np.arange(len(eps_muls))
xlabels = [str(e) for e in eps_muls]

# Panel (a): HM
ax1.plot(x, k3_hm, 'o-', color='#2166ac', linewidth=1.8, markersize=5, label='$K{=}3$ (ours)')
ax1.plot(x, k0_hm, 's--', color='#b2182b', linewidth=1.8, markersize=5, label='$K{=}0$ (no inner loop)')
ax1.axhspan(k3_hm.min() - 0.01, k3_hm.max() + 0.01, xmin=0, xmax=0.22,
            alpha=0.08, color='#2166ac', zorder=0)
ax1.axhline(k3_hm.mean(), color='#2166ac', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.axhline(k0_hm.mean(), color='#b2182b', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.text(7.1, k3_hm.mean() + 0.002, f'avg={k3_hm.mean():.3f}', fontsize=7.5, color='#2166ac', va='bottom')
ax1.text(7.1, k0_hm.mean() - 0.002, f'avg={k0_hm.mean():.3f}', fontsize=7.5, color='#b2182b', va='top')
ax1.annotate('Tight\nconstraints', xy=(0.5, 0.48), fontsize=8, color='#2166ac',
             fontstyle='italic', ha='center')
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels, fontsize=9)
ax1.set_xlabel(r'Constraint tightness ($\epsilon$ multiplier) $\longrightarrow$ looser', fontsize=9.5)
ax1.set_ylabel('Composite Score (HM $\\uparrow$)', fontsize=9.5)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_ylim(0.44, 0.57)
ax1.text(-0.08, 1.02, '(a)', transform=ax1.transAxes, fontsize=11, fontweight='bold')

# Panel (b): Retain quality
ax2.plot(x, k3_ret, 'o-', color='#2166ac', linewidth=1.8, markersize=5, label='$K{=}3$ (ours)')
ax2.plot(x, k0_ret, 's--', color='#b2182b', linewidth=1.8, markersize=5, label='$K{=}0$ (no inner loop)')
ax2.axhspan(k3_ret.min() - 0.01, k3_ret.max() + 0.01, xmin=0, xmax=0.22,
            alpha=0.08, color='#2166ac', zorder=0)
ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, fontsize=9)
ax2.set_xlabel(r'Constraint tightness ($\epsilon$ multiplier) $\longrightarrow$ looser', fontsize=9.5)
ax2.set_ylabel('Retain Quality ($r_k$ $\\uparrow$)', fontsize=9.5)
ax2.legend(fontsize=9, loc='lower right')
ax2.set_ylim(0.44, 0.53)
ax2.text(-0.08, 1.02, '(b)', transform=ax2.transAxes, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/design/figures/fig_inner_loop_ablation.pdf', bbox_inches='tight', dpi=300)
plt.savefig('docs/design/figures/fig_inner_loop_ablation.png', bbox_inches='tight', dpi=200)
print("Saved fig_inner_loop_ablation.pdf and .png")
