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
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
})

SAVES = Path("/data/open-unlearning/saves/unlearn")
OUT_DIR = Path("/data/open-unlearning/paper/figures")
OUT_DIR_DOCS = Path("/data/open-unlearning/docs/design/figures")

COLOR_FGT = '#d62728'
COLOR_RET = '#1f77b4'
COLOR_LAM = '#9467bd'
COLOR_EPS = '#ff7f0e'


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


def plot_pdu(ax, trainer_state_path, title):
    """Plot PDU dynamics on a single axes with dual y-axis."""
    with open(trainer_state_path) as f:
        d = json.load(f)

    logs = [e for e in d['log_history'] if 'forget_loss' in e]
    n = len(logs)
    steps = list(range(n))
    fgt = [e['forget_loss'] for e in logs]
    ret = [e['retain_loss'] for e in logs]

    ax.set_xlabel("Step")
    ax.set_ylabel(r'$\mathcal{L}_{\mathrm{forget}}$', color=COLOR_FGT)
    l1, = ax.plot(steps, fgt, color=COLOR_FGT, linewidth=1.8,
                  label=r'$\mathcal{L}_{\mathrm{forget}}$')
    ax.tick_params(axis='y', labelcolor=COLOR_FGT)
    ax.set_ylim(-20, max(fgt) * 1.1)

    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\mathcal{L}_{\mathrm{retain}}$', color=COLOR_RET)
    l2, = ax2.plot(steps, ret, color=COLOR_RET, linewidth=1.5,
                   label=r'$\mathcal{L}_{\mathrm{retain}}$')
    ax2.tick_params(axis='y', labelcolor=COLOR_RET)
    ax2.set_ylim(-0.02, max(ret) * 1.3)

    ax.legend([l1, l2], [l.get_label() for l in [l1, l2]],
              loc='upper right', framealpha=0.95)
    ax.set_title(title)


# ── MUSE Books: BLADE and PDU (for paper Fig 4) ───────────────────────────

blade_books_path = SAVES / "muse_Llama-2-7b-hf_Books_adaptive_T250_s42/lora_bial_history.json"
pdu_books_path = SAVES / "muse_Llama-2-7b-hf_Books_PDU_s42/trainer_state.json"

# Individual subplot figures (used in results.tex)
fig, ax = plt.subplots(figsize=(7, 3.2))
plot_blade(ax, blade_books_path, "MUSE Books — Llama-2-7B: BLADE, 78 Steps (HM = 0.823)")
plt.tight_layout()
for outdir in [OUT_DIR, OUT_DIR_DOCS]:
    fig.savefig(outdir / "fig_muse_books_dynamics.pdf", bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.2))
plot_pdu(ax, pdu_books_path, "MUSE Books — Llama-2-7B: PDU (HM = 0.602)")
plt.tight_layout()
for outdir in [OUT_DIR, OUT_DIR_DOCS]:
    fig.savefig(outdir / "fig_muse_books_PDU_dynamics.pdf", bbox_inches='tight', pad_inches=0.02)
plt.close(fig)

print("Done: fig_muse_books_dynamics.pdf + fig_muse_books_PDU_dynamics.pdf")


# ── TOFU 1B forget01: BLADE and PDU ──────────────────────────────────────

tofu_blade_paths = list(SAVES.glob("*1b_verify*s42*/lora_bial_history.json"))
tofu_pdu_paths = list(SAVES.glob("*tofu*1B*PDU*forget01*/trainer_state.json"))

if not tofu_blade_paths:
    tofu_blade_paths = list(SAVES.glob("*tofu*1B*forget01*adaptive*/lora_bial_history.json"))
    if not tofu_blade_paths:
        tofu_blade_paths = list(SAVES.glob("*tofu*1B*forget01*LoRABiAL*/lora_bial_history.json"))

if tofu_blade_paths:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    plot_blade(ax, tofu_blade_paths[0],
               "TOFU 1B fgt01 — BLADE (HM = 0.811)")
    plt.tight_layout()
    for outdir in [OUT_DIR, OUT_DIR_DOCS]:
        fig.savefig(outdir / "fig_tofu_1B_forget01_dynamics.pdf",
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Done: fig_tofu_1B_forget01_dynamics.pdf (from {tofu_blade_paths[0]})")
else:
    print("WARN: No TOFU 1B BLADE history found")

if tofu_pdu_paths:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    plot_pdu(ax, tofu_pdu_paths[0],
             "TOFU 1B fgt01 — PDU (HM = 0.690)")
    plt.tight_layout()
    for outdir in [OUT_DIR, OUT_DIR_DOCS]:
        fig.savefig(outdir / "fig_tofu_1B_forget01_PDU_dynamics.pdf",
                    bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Done: fig_tofu_1B_forget01_PDU_dynamics.pdf (from {tofu_pdu_paths[0]})")
else:
    print("WARN: No TOFU 1B PDU history found")


# ── Generate all other dynamics figures for appendix ──────────────────────

def find_blade_history(pattern):
    paths = list(SAVES.glob(pattern))
    return paths[0] if paths else None

def find_pdu_state(pattern):
    paths = list(SAVES.glob(pattern))
    return paths[0] if paths else None


appendix_configs = [
    # TOFU 1B
    ("fig_tofu_1B_forget05_dynamics.pdf",
     "*tofu*1B*forget05*adaptive*/lora_bial_history.json",
     "TOFU 1B fgt05 — BLADE", "blade"),
    ("fig_tofu_1B_forget05_PDU_dynamics.pdf",
     "*tofu*1B*PDU*forget05*/trainer_state.json",
     "TOFU 1B fgt05 — PDU", "pdu"),
    ("fig_tofu_1B_forget10_dynamics.pdf",
     "*tofu*1B*forget10*adaptive*/lora_bial_history.json",
     "TOFU 1B fgt10 — BLADE", "blade"),
    ("fig_tofu_1B_forget10_PDU_dynamics.pdf",
     "*tofu*1B*PDU*forget10*/trainer_state.json",
     "TOFU 1B fgt10 — PDU", "pdu"),
    # TOFU 3B
    ("fig_tofu_3B_forget01_dynamics.pdf",
     "*tofu*3B*forget01*adaptive*/lora_bial_history.json",
     "TOFU 3B fgt01 — BLADE", "blade"),
    ("fig_tofu_3B_forget01_PDU_dynamics.pdf",
     "*tofu*3B*PDU*forget01*/trainer_state.json",
     "TOFU 3B fgt01 — PDU", "pdu"),
    ("fig_tofu_3B_forget05_dynamics.pdf",
     "*tofu*3B*forget05*adaptive*/lora_bial_history.json",
     "TOFU 3B fgt05 — BLADE", "blade"),
    ("fig_tofu_3B_forget05_PDU_dynamics.pdf",
     "*tofu*3B*PDU*forget05*/trainer_state.json",
     "TOFU 3B fgt05 — PDU", "pdu"),
    ("fig_tofu_3B_forget10_dynamics.pdf",
     "*tofu*3B*forget10*adaptive*/lora_bial_history.json",
     "TOFU 3B fgt10 — BLADE", "blade"),
    ("fig_tofu_3B_forget10_PDU_dynamics.pdf",
     "*tofu*3B*PDU*forget10*/trainer_state.json",
     "TOFU 3B fgt10 — PDU", "pdu"),
    # MUSE News
    ("fig_muse_news_dynamics.pdf",
     "*News_adaptive*s42*/lora_bial_history.json",
     "MUSE News — BLADE", "blade"),
    ("fig_muse_news_PDU_dynamics.pdf",
     "*News_PDU*s42*/trainer_state.json",
     "MUSE News — PDU", "pdu"),
]

for fname, pattern, title, method in appendix_configs:
    path = find_blade_history(pattern) if method == "blade" else find_pdu_state(pattern)
    if path is None:
        print(f"SKIP: {fname} — no data at {pattern}")
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
