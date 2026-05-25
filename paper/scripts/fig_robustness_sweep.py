"""Generate robustness sweep figures for BLADE appendix.

Figure 1: HM bar chart with mean ± range across sweep values for each param.
Figure 2: Lambda trajectory panel showing how lambda evolves under each sweep param.

Data: TOFU 1B forget01, KnowUnDo privacy, MUSE Books — all K=3, 40 runs each.
"""

import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 600,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.4,
})

SAVES = Path("/data/open-unlearning/saves/unlearn")
OUT_DIR = Path("/data/open-unlearning/paper/figures")

# ── Data ──────────────────────────────────────────────────────────────────

# TOFU 1B forget01 K=3 HM values (from sweep_tofu_1b_01.md)
TOFU_HM = {
    r'$\epsilon_{\mathrm{mul}}$': {
        'values': [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.2],
        'HM': [0.8098, 0.8097, 0.8099, 0.8089, 0.8075, 0.8114, 0.8133, 0.8108],
        'range': '0.75–3.2',
    },
    r'$\tau$': {
        'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0],
        'HM': [0.7682, 0.8057, 0.8112, 0.8114, 0.8121, 0.8121, 0.8130, 0.8073],
        'range': '0.1–1.0',
    },
    r'$\alpha$': {
        'values': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        'HM': [0.8078, 0.8088, 0.8080, 0.8076, 0.8067, 0.8083, 0.8085, 0.8089],
        'range': '0.01–1.0',
    },
    r'$\rho$': {
        'values': [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        'HM': [0.8049, 0.8089, 0.8081, 0.8080, 0.8057, 0.8055, 0.8066, 0.8049],
        'range': '0.01–2.0',
    },
    r'$\eta_{\mathrm{in}}$': {
        'values': ['1e-5', '2.5e-5', '5e-5', '1e-4', '2e-4', '5e-4', '1e-3', '2e-3'],
        'HM': [0.8076, 0.8100, 0.8122, 0.8089, 0.8080, 0.8046, 0.7905, 0.7970],
        'range': '1e-5–2e-3',
    },
}

# KnowUnDo privacy K=3 HM values (from sweep_knowundo_privacy/all_results.csv)
KNOWUNDO_HM = {
    r'$\epsilon_{\mathrm{mul}}$': {
        'HM': [0.6012, 0.6125, 0.6021, 0.6035, 0.5908, 0.5787, 0.5890, 0.5861],
    },
    r'$\tau$': {
        'HM': [0.5362, 0.5390, 0.5629, 0.5782, 0.5912, 0.5935, 0.5713, 0.5913],
    },
    r'$\alpha$': {
        'HM': [0.5910, 0.6036, 0.5861, 0.6052, 0.5937, 0.5942, 0.5694, 0.5425],
    },
    r'$\rho$': {
        'HM': [0.6000, 0.6008, 0.6017, 0.5861, 0.6011, 0.6040, 0.5618, 0.6051],
    },
    r'$\eta_{\mathrm{in}}$': {
        'HM': [0.6051, 0.6007, 0.5962, 0.5855, 0.5861, 0.5880, 0.5844, 0.5605],
    },
}

# MUSE Books K=3 HM values (from sweep_muse_books/all_results.csv)
# Ordered by sweep parameter value: 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.2
MUSE_BOOKS_HM = {
    r'$\epsilon_{\mathrm{mul}}$': {
        'HM': [0.7501, 0.7987, 0.8125, 0.8144, 0.8018, 0.8014, 0.8164, 0.8207],
    },
    r'$\tau$': {
        'HM': [0.7583, 0.8007, 0.7805, 0.8205, 0.8079, 0.8042, 0.7960, 0.7923],
    },
    r'$\alpha$': {
        'HM': [0.7906, 0.8257, 0.8194, 0.8228, 0.8230, 0.8317, 0.8237, 0.8115],
    },
    r'$\rho$': {
        'HM': [0.7763, 0.7847, 0.8281, 0.8308, 0.8134, 0.3789, 0.7991, 0.7943],
    },
    r'$\eta_{\mathrm{in}}$': {
        'HM': [0.8243, 0.8226, 0.8216, 0.8280, 0.8254, 0.8131, 0.8056, 0.7654],
    },
}

# Lambda trajectory file paths (TOFU K=3)
LAMBDA_PATHS = {
    r'$\epsilon_{\mathrm{mul}}$': ('eps_mul', ['0.75', '1.5', '3.2']),
    r'$\tau$': ('tau', ['0.1', '0.5', '1.0']),
    r'$\alpha$': ('alpha_dual', ['0.01', '0.1', '1.0']),
    r'$\rho$': ('rho', ['0.01', '0.1', '2.0']),
    r'$\eta_{\mathrm{in}}$': ('eta_in', ['1e-5', '5e-4', '2e-3']),
}


# ── Figure 1: HM Robustness — one panel per param, lines per benchmark ───

# Sweep x-axis values (shared across benchmarks)
SWEEP_X = {
    r'$\epsilon_{\mathrm{mul}}$': [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.2],
    r'$\tau$': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.0],
    r'$\alpha$': [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    r'$\rho$': [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    r'$\eta_{\mathrm{in}}$': [1e-5, 2.5e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
}

BENCH_COLORS = {
    'TOFU': '#4e79a7',
    'KnowUnDo Priv.': '#e15759',
    'MUSE Books': '#59a14f',
}


def fig_hm_robustness():
    params = list(TOFU_HM.keys())
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2), sharey=True)

    for idx, param in enumerate(params):
        ax = axes[idx]
        x_vals = SWEEP_X[param]
        x = np.arange(len(x_vals))

        # TOFU
        tofu_hm = np.array(TOFU_HM[param]['HM'])
        ax.plot(x, tofu_hm, marker='o', markersize=3.5, color=BENCH_COLORS['TOFU'],
                linewidth=1.3, label='TOFU')

        # KnowUnDo Privacy
        ku_hm = np.array(KNOWUNDO_HM[param]['HM'])
        ax.plot(x, ku_hm, marker='s', markersize=3.5, color=BENCH_COLORS['KnowUnDo Priv.'],
                linewidth=1.3, label='KnowUnDo Priv.')

        # MUSE Books
        muse_hm = np.array(MUSE_BOOKS_HM[param]['HM'])
        ax.plot(x, muse_hm, marker='^', markersize=3.5, color=BENCH_COLORS['MUSE Books'],
                linewidth=1.3, label='MUSE Books')

        # x-axis
        ax.set_xticks(x)
        if param == r'$\eta_{\mathrm{in}}$':
            ax.set_xticklabels(['1e-5', '', '5e-5', '1e-4', '2e-4', '', '1e-3', '2e-3'],
                               fontsize=6.5, rotation=30, ha='right')
        else:
            ax.set_xticklabels([str(v) for v in x_vals], fontsize=7, rotation=30, ha='right')

        ax.set_title(param, fontsize=10)
        if idx == 0:
            ax.set_ylabel('HM')

        ax.set_ylim(0.3, 0.9)

        if idx == 4:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.7, handlelength=1.2)

        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        ax.grid(axis='y', alpha=0.2, linewidth=0.3)

    fig.tight_layout(w_pad=1.5)
    out = OUT_DIR / 'fig_robustness_hm.pdf'
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ── Figure 2: Lambda Trajectory Panel (2 rows: TOFU + KnowUnDo) ──────────

# KnowUnDo Privacy lambda paths (prefix: privacy_k3_sweep_)
KNOWUNDO_LAMBDA_PATHS = {
    r'$\epsilon_{\mathrm{mul}}$': ('eps_mul', ['0.75', '1.5', '3.2']),
    r'$\tau$': ('tau', ['0.1', '0.5', '1.0']),
    r'$\alpha$': ('alpha_dual', ['0.01', '0.2', '1.0']),
    r'$\rho$': ('rho', ['0.01', '0.1', '2.0']),
    r'$\eta_{\mathrm{in}}$': ('eta_in', ['1e-5', '2e-4', '2e-3']),
}

# MUSE Books lambda paths (prefix: books_k3_sweep_)
MUSE_BOOKS_LAMBDA_PATHS = {
    r'$\epsilon_{\mathrm{mul}}$': ('eps_mul', ['0.75', '1.5', '3.2']),
    r'$\tau$': ('tau', ['0.1', '0.5', '1.0']),
    r'$\alpha$': ('alpha_dual', ['0.01', '0.5', '1.0']),
    r'$\rho$': ('rho', ['0.01', '0.1', '2.0']),
    r'$\eta_{\mathrm{in}}$': ('eta_in', ['1e-5', '2e-4', '2e-3']),
}


def fig_lambda_trajectories():
    fig, axes = plt.subplots(3, 5, figsize=(10, 5.4), sharey=False)

    cmap = ['#1f77b4', '#ff7f0e', '#d62728']

    # Row 0: TOFU
    for idx, (param_label, (param_key, selected_vals)) in enumerate(LAMBDA_PATHS.items()):
        ax = axes[0, idx]

        for i, val in enumerate(selected_vals):
            path = SAVES / f'sweep_{param_key}_{val}' / 'lora_bial_history.json'
            if not path.exists():
                continue
            with open(path) as f:
                h = json.load(f)
            steps = [r['step'] for r in h]
            lam = [r['lambda'] for r in h]
            ax.plot(steps, lam, color=cmap[i], linewidth=1.3,
                    label=f'{val}', alpha=0.9)

        if idx == 0:
            ax.set_ylabel(r'$\lambda$ (TOFU)')
        ax.set_title(param_label, fontsize=10)
        ax.legend(loc='best', framealpha=0.7, handlelength=1.2, fontsize=7)

        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        ax.grid(alpha=0.2, linewidth=0.3)

    # Row 1: KnowUnDo Privacy
    for idx, (param_label, (param_key, selected_vals)) in enumerate(KNOWUNDO_LAMBDA_PATHS.items()):
        ax = axes[1, idx]

        for i, val in enumerate(selected_vals):
            path = SAVES / f'privacy_k3_sweep_{param_key}_{val}' / 'lora_bial_history.json'
            if not path.exists():
                continue
            with open(path) as f:
                h = json.load(f)
            steps = [r['step'] for r in h]
            lam = [r['lambda'] for r in h]
            ax.plot(steps, lam, color=cmap[i], linewidth=1.3,
                    label=f'{val}', alpha=0.9)

        if idx == 0:
            ax.set_ylabel(r'$\lambda$ (KU Priv.)')
        ax.legend(loc='best', framealpha=0.7, handlelength=1.2, fontsize=7)

        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        ax.grid(alpha=0.2, linewidth=0.3)

    # Row 2: MUSE Books
    for idx, (param_label, (param_key, selected_vals)) in enumerate(MUSE_BOOKS_LAMBDA_PATHS.items()):
        ax = axes[2, idx]

        for i, val in enumerate(selected_vals):
            path = SAVES / f'books_k3_sweep_{param_key}_{val}' / 'lora_bial_history.json'
            if not path.exists():
                continue
            with open(path) as f:
                h = json.load(f)
            steps = [r['step'] for r in h]
            lam = [r['lambda'] for r in h]
            ax.plot(steps, lam, color=cmap[i], linewidth=1.3,
                    label=f'{val}', alpha=0.9)

        ax.set_xlabel('Step')
        if idx == 0:
            ax.set_ylabel(r'$\lambda$ (MUSE Books)')
        ax.legend(loc='best', framealpha=0.7, handlelength=1.2, fontsize=7)

        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)
        ax.grid(alpha=0.2, linewidth=0.3)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    out = OUT_DIR / 'fig_robustness_lambda.pdf'
    fig.savefig(out, dpi=600, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    fig_hm_robustness()
    fig_lambda_trajectories()
    print('Done.')
