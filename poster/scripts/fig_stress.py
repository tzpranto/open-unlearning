"""Poster version of paper/scripts/fig_scalability_sustainability.py (same data, MUSE News HM).

Panels side by side, PSU colors, Roboto, drawn at print size for the poster's right column.
Run from the repo root:  python poster/scripts/fig_stress.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
for path in ['/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF',
             '/usr/share/texlive/texmf-dist/fonts/truetype/google/roboto']:
    if os.path.isdir(path):
        for f in font_manager.findSystemFonts(fontpaths=[path]):
            font_manager.fontManager.addfont(f)

BEAVER = '#1E407C'
PINK = '#BC204B'

plt.rcParams.update({
    'font.family': 'Roboto',
    'font.size': 21,
    'axes.labelsize': 21,
    'axes.titlesize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 21,
    'axes.linewidth': 1.6,
    'xtick.major.width': 1.6,
    'ytick.major.width': 1.6,
    'lines.linewidth': 3.5,
    'lines.markersize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Data from the paper (Scalability and Sustainability, MUSE News, Llama-2-7B)
x = [1, 2, 3, 4]
pdu_scal = [0.581, 0.579, 0.573, 0.005]
blade_scal = [0.542, 0.547, 0.552, 0.533]
pdu_sust = [0.580, 0.558, 0.016, 0.130]
blade_sust = [0.542, 0.545, 0.523, 0.525]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7), sharey=True)
panels = [
    (axes[0], pdu_scal, blade_scal, '(a) Scalability', 'Forget set size',
     [r'1$\times$' + '\n(889)', r'2$\times$' + '\n(1778)', r'3$\times$' + '\n(2667)',
      r'4$\times$' + '\n(3554)'], (4, 0.005)),
    (axes[1], pdu_sust, blade_sust, '(b) Sustainability', 'Sequential unlearning step',
     ['1', '2', '3', '4'], (3, 0.016)),
]
for ax, pdu, blade, title, xlabel, ticks, (cx, cy) in panels:
    ax.plot(x, blade, 'o-', color=BEAVER, label='BLADE', zorder=5)
    ax.plot(x, pdu, 's--', color=PINK, label='PDU', zorder=4)
    ax.set_title(title, fontweight='bold', color=BEAVER, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(-0.03, 0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis='y', alpha=0.35, linewidth=1.2)
    ax.annotate('collapse', xy=(cx, cy + 0.03), xytext=(cx - 1.5, 0.12),
                fontsize=21, color=PINK, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=PINK, lw=2.5, mutation_scale=24))
axes[0].set_ylabel('HM')
axes[0].legend(loc='lower left', frameon=False, bbox_to_anchor=(0.0, 0.2))

fig.tight_layout(w_pad=2)
out = os.path.join(HERE, '..', 'figures', 'fig_stress.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
print('Saved:', os.path.normpath(out))
