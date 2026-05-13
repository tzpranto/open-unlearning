"""LLM Judge results as grouped bar chart across all benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import os

methods = ["GradDiff", "NPO", "SimNPO", "RMU", "BLURNPO", "PDU", "BLADE"]

# HM_J values (mean, std)
tofu_1b_fgt01 = [0.620, 0.696, 0.288, 0.721, 0.541, 0.746, 0.929]
tofu_1b_fgt01_std = [0.02, 0.03, 0.03, 0.01, 0.03, 0.02, 0.01]

tofu_1b_fgt05 = [0.643, 0.731, 0.435, 0.736, 0.709, 0.850, 0.919]
tofu_1b_fgt05_std = [0.00, 0.01, 0.01, 0.01, 0.01, 0.01, 0.00]

tofu_1b_fgt10 = [0.625, 0.665, 0.435, 0.854, 0.255, 0.906, 0.919]
tofu_1b_fgt10_std = [0.01, 0.03, 0.02, 0.01, 0.21, 0.00, 0.01]

muse_books = [0.010, 0.647, 0.753, 0.791, 0.720, 0.586, 0.810]
muse_books_std = [0.01, 0.01, 0.02, 0.00, 0.02, 0.01, 0.01]

muse_news = [0.529, 0.531, 0.447, 0.565, 0.406, 0.638, 0.588]
muse_news_std = [0.03, 0.01, 0.01, 0.00, 0.04, 0.02, 0.01]

knowundo_copy = [0.41, 0.66, 0.75, 0.60, 0.52, 0.68, 0.78]
knowundo_copy_std = [0.04, 0.02, 0.01, 0.02, 0.07, 0.03, 0.01]

knowundo_priv = [0.62, 0.68, 0.77, 0.08, 0.61, 0.72, 0.83]
knowundo_priv_std = [0.04, 0.01, 0.01, 0.02, 0.05, 0.02, 0.02]

datasets = ["TOFU 1B\n(fgt01)", "TOFU 1B\n(fgt05)", "TOFU 1B\n(fgt10)", "MUSE\nBooks", "MUSE\nNews", "KnowUndo\nCopyright", "KnowUndo\nPrivacy"]
all_data = [tofu_1b_fgt01, tofu_1b_fgt05, tofu_1b_fgt10, muse_books, muse_news, knowundo_copy, knowundo_priv]
all_std = [tofu_1b_fgt01_std, tofu_1b_fgt05_std, tofu_1b_fgt10_std, muse_books_std, muse_news_std, knowundo_copy_std, knowundo_priv_std]

fig, ax = plt.subplots(figsize=(7, 3.5))

n_methods = len(methods)
n_datasets = len(datasets)
x = np.arange(n_datasets)
total_width = 0.75
bar_width = total_width / n_methods

colors = ['#8c8c8c', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d', '#feb24c', '#e31a1c']

for i, (method, color) in enumerate(zip(methods, colors)):
    vals = [all_data[d][i] for d in range(n_datasets)]
    stds = [all_std[d][i] for d in range(n_datasets)]
    offset = (i - n_methods/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, vals, bar_width, label=method, color=color,
                  alpha=1.0, edgecolor='white', linewidth=0.3)
    ax.errorbar(x + offset, vals, yerr=stds, fmt='none', ecolor='gray',
                elinewidth=0.5, capsize=1.5, capthick=0.5)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylabel("HM$_\\mathrm{J}$", fontsize=11)
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', fontsize=9, ncol=4, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_llm_judge.pdf')
plt.savefig(out_path, bbox_inches='tight', dpi=600)
print(f"Saved: {out_path}")
