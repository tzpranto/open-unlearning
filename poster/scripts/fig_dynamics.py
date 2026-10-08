"""Poster version of the paper's BLADE training dynamics on MUSE Books (seed 42), BLADE only.

Data: muse_books_blade_dynamics.json, read back from the paper figure fig_muse_books_dynamics.pdf
(the training log itself is not in this repo). PSU colors, Roboto, print size.
Run from the repo root:  python poster/scripts/fig_dynamics.py
"""

import json
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
NAVY = '#001E44'
TINT = '#EAF2FB'

plt.rcParams.update({
    'font.family': 'Roboto',
    'font.size': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,
    'axes.linewidth': 1.6,
    'xtick.major.width': 1.6,
    'ytick.major.width': 1.6,
    'axes.spines.top': False,
})

with open(os.path.join(HERE, 'muse_books_blade_dynamics.json')) as f:
    d = json.load(f)
steps, t = d['step'], d['t_trans']

fig, ax = plt.subplots(figsize=(9.6, 4.6))
ax2 = ax.twinx()

# phases: warm-up | spike and ratchet | convergence
ax.axvspan(t, 42, color=TINT, zorder=0)
for x0, x1, name in [(0, t, 'warm-up'), (t, 42, 'spike +\nratchet'), (42, 77, 'convergence')]:
    ax.text((x0 + x1) / 2, 8.25, name, ha='center', va='bottom', fontsize=21,
            fontweight='bold', color=BEAVER)

l1, = ax.plot(steps, d['L_fgt'], color=PINK, lw=4, label=r'$\mathcal{L}_{\mathrm{fgt}}$')
l2, = ax2.plot(steps, d['L_ret'], color=BEAVER, lw=3.5, label=r'$\mathcal{L}_{\mathrm{ret}}$')
l3, = ax2.plot(steps, d['lambda'], color=NAVY, lw=3.5, ls='--', label=r'$\lambda$')

ax.set_xlabel('step')
ax.set_xlim(0, 77)
ax.set_ylim(-0.2, 8.2)
ax.set_yticks([0, 4, 8])
ax.set_ylabel(r'$\mathcal{L}_{\mathrm{fgt}}$', color=PINK)
ax.tick_params(axis='y', colors=PINK)
ax2.set_ylim(-0.1, 4.1)
ax2.set_yticks([0, 2, 4])
ax2.set_ylabel(r'$\mathcal{L}_{\mathrm{ret}},\ \lambda$', color=BEAVER)
ax2.tick_params(axis='y', colors=BEAVER)
# direct curve labels instead of a legend
ax.text(12, 6.5, r'$\mathcal{L}_{\mathrm{fgt}}$', color=PINK, fontsize=24, ha='center', va='top')
ax2.text(62, 0.15, r'$\mathcal{L}_{\mathrm{ret}}$', color=BEAVER, fontsize=24, ha='center', va='bottom')
ax2.text(62, 2.6, r'$\lambda$', color=NAVY, fontsize=26, ha='center', va='bottom')

fig.tight_layout()
out = os.path.join(HERE, '..', 'figures', 'fig_dynamics.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.04)
print('Saved:', os.path.normpath(out))
