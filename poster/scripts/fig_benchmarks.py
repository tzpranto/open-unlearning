"""Poster dumbbell chart: strongest baseline vs BLADE (HM, 5 seeds) on every benchmark setting.

Numbers from the paper's TOFU 1B / TOFU 3B / MUSE / KnowUndo tables.
Run from the repo root:  python poster/scripts/fig_benchmarks.py
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
GREY = '#8C8C8C'

plt.rcParams.update({
    'font.family': 'Roboto',
    'font.size': 19,
    'axes.labelsize': 20,
    'xtick.labelsize': 15,
    'ytick.labelsize': 18,
    'legend.fontsize': 19,
    'axes.linewidth': 1.6,
    'xtick.major.width': 1.6,
    'ytick.major.width': 1.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# (setting, best baseline name, best baseline HM, BLADE HM)
rows = [
    ('1%', 'PDU', 0.690, 0.808),
    ('5%', 'PDU', 0.740, 0.800),
    ('10%', 'PDU', 0.797, 0.803),
    ('1%', 'PDU', 0.742, 0.846),
    ('5%', 'PDU', 0.843, 0.842),
    ('10%', 'PDU', 0.857, 0.836),
    ('Books', 'SimNPO', 0.753, 0.818),
    ('News', 'PDU', 0.577, 0.544),
    ('Copyright', 'SimNPO', 0.461, 0.477),
    ('Privacy', 'PDU', 0.546, 0.605),
]
groups = [(1, 'TOFU 1B'), (4, 'TOFU 3B'), (6.5, 'MUSE'), (8.5, 'KnowUndo')]

fig, ax = plt.subplots(figsize=(10.4, 3.0))
for i, (_, name, base, blade) in enumerate(rows):
    color = BEAVER if blade >= base else PINK
    ax.plot([i, i], [base, blade], color=color, lw=4, zorder=2, solid_capstyle='round')
    ax.scatter(i, base, s=170, facecolor='white', edgecolor=GREY, linewidth=3, zorder=3)
    ax.scatter(i, blade, s=200, color=BEAVER, zorder=4)
    ax.annotate(name, (i, min(base, blade) - 0.03), ha='center', va='top', fontsize=14, color=GREY)

ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r[0] for r in rows])
ax.set_xlim(-0.5, len(rows) - 0.5)
ax.set_ylim(0.33, 0.9)
ax.set_yticks([0.4, 0.6, 0.8])
ax.set_ylabel('HM')
ax.grid(axis='y', alpha=0.35, linewidth=1.2)
for x in (2.5, 5.5, 7.5):
    ax.axvline(x, color=GREY, lw=1, alpha=0.5)

for x, name in groups:
    ax.text(x, -0.27, name, transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=19, fontweight='bold', color=BEAVER)

fig.tight_layout()
out = os.path.join(HERE, '..', 'figures', 'fig_benchmarks.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
print('Saved:', os.path.normpath(out))
