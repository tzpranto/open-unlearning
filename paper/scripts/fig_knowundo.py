"""Generate KnowUndo grouped bar chart: HM and HM_J for copyright and privacy."""

import matplotlib.pyplot as plt
import numpy as np
import os

methods = ["GradDiff", "NPO", "SimNPO", "RMU", "BLURNPO", "PDU", "BLADE"]

# Copyright domain (mean, std)
hm_copyright = [0.348, 0.433, 0.461, 0.347, 0.353, 0.442, 0.477]
hm_copyright_std = [0.01, 0.00, 0.01, 0.01, 0.03, 0.01, 0.00]
hmj_copyright = [0.41, 0.66, 0.75, 0.60, 0.52, 0.68, 0.78]
hmj_copyright_std = [0.04, 0.02, 0.01, 0.02, 0.07, 0.03, 0.01]

# Privacy domain (mean, std)
hm_privacy = [0.513, 0.496, 0.544, 0.182, 0.498, 0.546, 0.605]
hm_privacy_std = [0.01, 0.01, 0.01, 0.01, 0.03, 0.01, 0.01]
hmj_privacy = [0.62, 0.68, 0.77, 0.08, 0.61, 0.72, 0.83]
hmj_privacy_std = [0.04, 0.01, 0.01, 0.02, 0.05, 0.02, 0.02]

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

x = np.arange(len(methods))
width = 0.35

colors_hm = '#4878CF'
colors_hmj = '#D65F5F'

for ax, hm, hm_std, hmj, hmj_std, title in [
    (axes[0], hm_copyright, hm_copyright_std, hmj_copyright, hmj_copyright_std, "Copyright"),
    (axes[1], hm_privacy, hm_privacy_std, hmj_privacy, hmj_privacy_std, "Privacy"),
]:
    bars1 = ax.bar(x - width/2, hm, width, label="HM (std.)", color=colors_hm, alpha=0.85)
    bars2 = ax.bar(x + width/2, hmj, width, label="HM$_\\mathrm{J}$ (judge)", color=colors_hmj, alpha=0.85)

    ax.errorbar(x - width/2, hm, yerr=hm_std, fmt='none', ecolor='black',
                elinewidth=0.7, capsize=2, capthick=0.7)
    ax.errorbar(x + width/2, hmj, yerr=hmj_std, fmt='none', ecolor='black',
                elinewidth=0.7, capsize=2, capthick=0.7)

    # Highlight BLADE
    bars1[-1].set_edgecolor('black')
    bars1[-1].set_linewidth(1.5)
    bars2[-1].set_edgecolor('black')
    bars2[-1].set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score" if ax == axes[0] else "")
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[1].legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_knowundo.pdf')
plt.savefig(out_path, bbox_inches='tight', dpi=600)
print(f"Saved: {out_path}")
