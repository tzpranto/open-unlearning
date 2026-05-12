import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
})

# Data from the tables
scales = [1, 2, 3, 4]
scale_labels = [r'1$\times$' + '\n(889)', r'2$\times$' + '\n(1778)', r'3$\times$' + '\n(2667)', r'4$\times$' + '\n(3554)']
steps = [1, 2, 3, 4]

# Scalability HM
pdu_scal = [0.581, 0.579, 0.573, 0.005]
blade_scal = [0.542, 0.547, 0.552, 0.533]

# Sustainability HM
pdu_sust = [0.580, 0.558, 0.016, 0.130]
blade_sust = [0.542, 0.545, 0.523, 0.525]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)

color_blade = '#2563EB'
color_pdu = '#DC2626'

# Left: Scalability
ax1.plot(scales, blade_scal, 'o-', color=color_blade, label='BLADE', zorder=5)
ax1.plot(scales, pdu_scal, 's--', color=color_pdu, label='PDU', zorder=5)
ax1.axhline(y=0.1, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
ax1.set_xlabel('Forget set scale (samples)')
ax1.set_ylabel('HM (harmonic mean)')
ax1.set_title('(a) Scalability', fontweight='medium')
ax1.set_xticks(scales)
ax1.set_xticklabels(scale_labels)
ax1.set_ylim(-0.02, 0.68)
ax1.set_yticks(np.arange(0, 0.7, 0.1))
ax1.legend(loc='lower left', framealpha=0.9, edgecolor='0.8')
ax1.grid(True, alpha=0.3, linestyle='-')
ax1.annotate('Collapse', xy=(4, 0.005), xytext=(3.4, 0.10),
             fontsize=7.5, color=color_pdu, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color=color_pdu, lw=1.0))

# Right: Sustainability
ax2.plot(steps, blade_sust, 'o-', color=color_blade, label='BLADE', zorder=5)
ax2.plot(steps, pdu_sust, 's--', color=color_pdu, label='PDU', zorder=5)
ax2.axhline(y=0.1, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
ax2.set_xlabel('Sequential unlearning step')
ax2.set_title('(b) Sustainability', fontweight='medium')
ax2.set_xticks(steps)
ax2.set_xticklabels(['1', '2', '3', '4'])
ax2.legend(loc='lower left', framealpha=0.9, edgecolor='0.8')
ax2.grid(True, alpha=0.3, linestyle='-')
ax2.annotate('Collapse', xy=(3, 0.016), xytext=(2.4, 0.10),
             fontsize=7.5, color=color_pdu, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color=color_pdu, lw=1.0))

plt.tight_layout(w_pad=1.5)
plt.savefig('paper/figures/fig_scalability_sustainability.pdf',
            bbox_inches='tight', pad_inches=0.02)
plt.close()
print("Done: figures/fig_scalability_sustainability.pdf")
