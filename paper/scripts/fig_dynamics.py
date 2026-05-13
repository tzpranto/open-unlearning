"""Generate consistent BLADE vs PDU dynamics figures for the paper (Fig 4)."""

import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.8,
})

SAVES = Path("/data/open-unlearning/saves/unlearn")
OUT_DIR = Path("/data/open-unlearning/paper/figures")
OUT_DIR_DOCS = Path("/data/open-unlearning/docs/design/figures")

COLOR_FGT = '#d62728'
COLOR_RET = '#1f77b4'
COLOR_LAM = '#9467bd'
COLOR_EPS = '#ff7f0e'


def style_axes(ax):
    """Set consistent, light spine styling across all subplots."""
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)


def plot_blade(ax, history_path, title):
    """Plot BLADE dynamics with dual y-axis: forget (left, red), retain (right, blue)."""
    with open(history_path) as f:
        h = json.load(f)

    steps = [r['step'] for r in h]
    lfgt = [r['L_fgt'] for r in h]
    lret = [r['L_ret'] for r in h]
    lam = [r['lambda'] for r in h]
    t_trans_vals = [r['t_trans'] for r in h if r['t_trans'] is not None]
    t_trans = t_trans_vals[0] if t_trans_vals else None

    ax.set_xlabel("Step")
    ax.set_ylabel(r'$\mathcal{L}_{\mathrm{forget}}$', color=COLOR_FGT)
    l1, = ax.plot(steps, lfgt, color=COLOR_FGT, linewidth=1.8,
                  label=r'$\mathcal{L}_{\mathrm{forget}}$')
    ax.tick_params(axis='y', labelcolor=COLOR_FGT)
    ax.set_ylim(-0.2, max(lfgt) * 1.15)

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\mathcal{L}_{\mathrm{retain}}$', color=COLOR_RET)
    l2, = ax2.plot(steps, lret, color=COLOR_RET, linewidth=1.5,
                   label=r'$\mathcal{L}_{\mathrm{retain}}$')
    l3, = ax2.plot(steps, lam, color=COLOR_LAM, linewidth=1.5, linestyle='--',
                   label=r'$\lambda$')
    ax2.tick_params(axis='y', labelcolor=COLOR_RET)
    ax2.set_ylim(-0.1, max(max(lret), max(lam)) * 1.15)

    if t_trans is not None:
        ax.axvline(t_trans, color='green', linestyle=':', linewidth=1.0,
                   alpha=0.6)
        ax.text(t_trans + 0.5, max(lfgt) * 1.05,
                r'$T_{\mathrm{trans}}$', color='green', fontsize=8, alpha=0.7)

    ax.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
              loc='upper right', framealpha=0.95)
    ax.set_title(title)
    style_axes(ax)
    style_axes(ax2)


def plot_pdu(ax, trainer_state_path, title, max_points=400):
    """Plot PDU dynamics on a single axes with dual y-axis."""
    with open(trainer_state_path) as f:
        d = json.load(f)

    logs = [e for e in d['log_history'] if 'forget_loss' in e]
    n = len(logs)
    # Subsample if too many points (keeps plot readable at small sizes)
    if n > max_points:
        stride = n // max_points
        logs = logs[::stride]
        n = len(logs)
    steps = list(range(n))
    fgt = [e['forget_loss'] for e in logs]
    ret = [e['retain_loss'] for e in logs]

    ax.set_xlabel("Step")
    ax.set_ylabel(r'$\mathcal{L}_{\mathrm{forget}}$', color=COLOR_FGT)
    l1, = ax.plot(steps, fgt, color=COLOR_FGT, linewidth=1.8,
                  label=r'$\mathcal{L}_{\mathrm{forget}}$')
    ax.tick_params(axis='y', labelcolor=COLOR_FGT)
    ax.set_ylim(-0.5, max(fgt) * 1.1)

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\mathcal{L}_{\mathrm{retain}}$', color=COLOR_RET)
    l2, = ax2.plot(steps, ret, color=COLOR_RET, linewidth=1.5,
                   label=r'$\mathcal{L}_{\mathrm{retain}}$')
    ax2.tick_params(axis='y', labelcolor=COLOR_RET)
    ax2.set_ylim(-0.02, max(ret) * 1.3)

    ax.legend([l1, l2], [l.get_label() for l in [l1, l2]],
              loc='upper right', framealpha=0.95)
    ax.set_title(title)
    style_axes(ax)
    style_axes(ax2)


# ── Data paths ───────────────────────────────────────────────────────────
BACKUP = Path("/data/open-unlearning-h100-backup/saves/unlearn")


# ── Main paper Fig 4: MUSE Books BLADE + PDU ─────────────────────────────

blade_books_path = SAVES / "muse_Llama-2-7b-hf_Books_adaptive_T250_s42/lora_bial_history.json"
pdu_books_path = SAVES / "muse_Llama-2-7b-hf_Books_PDU_s42/trainer_state.json"

fig, ax = plt.subplots(figsize=(7, 3.2))
plot_blade(ax, blade_books_path, "MUSE Books — BLADE (HM = 0.823)")
plt.tight_layout()
for outdir in [OUT_DIR, OUT_DIR_DOCS]:
    fig.savefig(outdir / "fig_muse_books_dynamics.pdf", bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.2))
plot_pdu(ax, pdu_books_path, "MUSE Books — PDU (HM = 0.602)")
plt.tight_layout()
for outdir in [OUT_DIR, OUT_DIR_DOCS]:
    fig.savefig(outdir / "fig_muse_books_PDU_dynamics.pdf", bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

print("Done: fig_muse_books_dynamics.pdf + fig_muse_books_PDU_dynamics.pdf")


# ── Appendix: 4 BLADE + 4 PDU (TOFU 1B, MUSE News, KnowUndo copy/priv) ──

appendix_configs = [
    # BLADE
    ("fig_tofu_1B_dynamics.pdf",
     BACKUP / "adaptive_Llama-3.2-1B-Instruct_forget01_s42/lora_bial_history.json",
     "TOFU 1B fgt01 — BLADE", "blade"),
    ("fig_muse_news_dynamics.pdf",
     SAVES / "muse_Llama-2-7b-hf_News_adaptive_s42/lora_bial_history.json",
     "MUSE News — BLADE", "blade"),
    ("fig_knowundo_copyright_dynamics.pdf",
     SAVES / "knowundo_BLADE_copyright_unified_s42/lora_bial_history.json",
     "KnowUndo Copyright — BLADE", "blade"),
    ("fig_knowundo_privacy_dynamics.pdf",
     SAVES / "knowundo_BLADE_privacy_unified_s42/lora_bial_history.json",
     "KnowUndo Privacy — BLADE", "blade"),
    # PDU
    ("fig_tofu_1B_PDU_dynamics.pdf",
     SAVES / "bs32_Llama-3.2-1B-Instruct_forget01_PDU_s42/trainer_state.json",
     "TOFU 1B fgt01 — PDU", "pdu"),
    ("fig_muse_news_PDU_dynamics.pdf",
     SAVES / "muse_Llama-2-7b-hf_News_PDU_s42/trainer_state.json",
     "MUSE News — PDU", "pdu"),
    ("fig_knowundo_copyright_PDU_dynamics.pdf",
     SAVES / "knowundo_copyright_pdu_s42/trainer_state.json",
     "KnowUndo Copyright — PDU", "pdu"),
    ("fig_knowundo_privacy_PDU_dynamics.pdf",
     SAVES / "knowundo_privacy_pdu_s42/trainer_state.json",
     "KnowUndo Privacy — PDU", "pdu"),
]

for fname, path, title, method in appendix_configs:
    if not path.exists():
        print(f"SKIP: {fname} — no data at {path}")
        continue
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if method == "blade":
        plot_blade(ax, path, title)
    else:
        plot_pdu(ax, path, title)
    plt.tight_layout()
    for outdir in [OUT_DIR, OUT_DIR_DOCS]:
        fig.savefig(outdir / fname, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Done: {fname}")

print("\nAll dynamics figures generated.")
