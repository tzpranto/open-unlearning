"""Poster grouped bar chart: HM of every method on every benchmark setting (5 seeds).

Numbers from the paper's TOFU 1B / TOFU 3B / MUSE / KnowUndo tables (HM column).
Headline gains are the mean, over a family's settings, of BLADE's relative gain vs the best baseline.
Drawn at print size for the full-width Experiments row.
Run from the repo root:  python poster/scripts/fig_hm_bars.py
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
for path in ['/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF',
             '/usr/share/texlive/texmf-dist/fonts/truetype/google/roboto']:
    if os.path.isdir(path):
        for f in font_manager.findSystemFonts(fontpaths=[path]):
            font_manager.fontManager.addfont(f)

BEAVER = '#1E407C'
GREY = '#8C8C8C'

plt.rcParams.update({
    'font.family': 'Roboto',
    'font.size': 26,
    'axes.labelsize': 28,
    'xtick.labelsize': 26,
    'ytick.labelsize': 24,
    'legend.fontsize': 26,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Baselines in greys, constraint-based baselines in Pugh tints, BLADE in Beaver Blue.
methods = ['GradAscent', 'GradDiff', 'NPO', 'SimNPO', 'RMU', 'BLURNPO', 'PDU', 'BLADE']
colors = ['#E3E3E3', '#C9C9C9', '#AEAEAE', '#939393', '#787878', '#CFE0F3', '#96BEE6', BEAVER]

# (tick label, HM per method in the order above)
settings = [
    ('1%',        [.553, .564, .560, .240, .576, .417, .690, .808]),
    ('5%',        [.022, .614, .603, .248, .583, .541, .740, .800]),
    ('10%',       [.000, .617, .590, .255, .691, .154, .797, .803]),
    ('1%',        [.509, .483, .525, .151, .246, .253, .742, .846]),
    ('5%',        [.632, .653, .640, .175, .483, .412, .843, .842]),
    ('10%',       [.000, .662, .640, .191, .591, .502, .857, .836]),
    ('Books',     [.000, .011, .638, .753, .738, .742, .600, .818]),
    ('News',      [.009, .479, .522, .440, .531, .478, .577, .544]),
    ('Copyright', [.000, .348, .433, .461, .347, .353, .442, .477]),
    ('Privacy',   [.017, .513, .496, .544, .182, .498, .546, .605]),
]
# (center x, label) for benchmark families, and (x span, headline) for gains
families = [(1, 'TOFU · Llama-3.2-1B'), (4, 'TOFU · Llama-3.2-3B'),
            (6.5, 'MUSE · Llama-2-7B'), (8.5, 'KnowUndo · Llama-2-7B-chat')]
gains = [((-0.4, 5.4), '+6%'), ((5.6, 6.4), '+9%'), ((7.6, 9.4), '+7%')]

n = len(methods)
x = np.arange(len(settings))
width = 0.86 / n

fig, ax = plt.subplots(figsize=(30.5, 7.8))
for i, (method, color) in enumerate(zip(methods, colors)):
    vals = [s[1][i] for s in settings]
    ax.bar(x + (i - n / 2 + 0.5) * width, vals, width, color=color, label=method,
           edgecolor='white', linewidth=0.8, zorder=3)
for j, (_, vals) in enumerate(settings):
    xb = x[j] + (n / 2 - 0.5) * width
    ax.text(xb, vals[-1] + 0.015, f'{vals[-1]:.2f}'.lstrip('0'), ha='center', va='bottom',
            fontsize=21, fontweight='bold', color=BEAVER)

ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in settings])
ax.tick_params(axis='x', length=0, pad=10)
ax.set_xlim(-0.5, len(settings) - 0.5)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_ylabel('HM (harmonic mean)')
ax.grid(axis='y', alpha=0.35, linewidth=1.5, zorder=0)
for xv in (2.5, 5.5, 7.5):
    ax.axvline(xv, color=GREY, lw=1.5, alpha=0.6)

for xc, name in families:
    ax.text(xc, -0.17, name, transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=28, fontweight='bold', color=BEAVER)
for (x0, x1), label in gains:
    ax.plot([x0, x1], [0.99, 0.99], color=BEAVER, lw=3, solid_capstyle='butt')
    ax.text((x0 + x1) / 2, 1.0, label, ha='center', va='bottom', fontsize=40,
            fontweight='bold', color=BEAVER)

# legend is typeset in LaTeX (blade-poster.tex) so it can carry citations
fig.tight_layout()
out = os.path.join(HERE, '..', 'figures', 'fig_hm_bars.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
print('Saved:', os.path.normpath(out))
