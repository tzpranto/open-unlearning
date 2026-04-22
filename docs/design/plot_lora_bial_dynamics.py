"""Generate loss dynamics figures for the LoRA-BiAL architecture report."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

SAVES = Path("/datadrive/forked/open-unlearning/saves")


def load_history(path):
    with open(path) as f:
        return json.load(f)


def extract(history, key):
    return [r[key] for r in history]


# ── Load data ──────────────────────────────────────────────────────────────
h18 = load_history(SAVES / "unlearn/muse_books_exp_18/lora_bial_history.json")
h_tofu = load_history(SAVES / "unlearn/tofu_Llama-3.2-1B-Instruct_forget01_LoRABiAL_T150/lora_bial_history.json")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: MUSE Books (Exp 18) — combined loss + λ
# ═══════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 4.5))

steps = extract(h18, "step")
lfgt = extract(h18, "L_fgt")
lret = extract(h18, "L_ret")
lam = extract(h18, "lambda")
epochs = extract(h18, "epoch")

# Epoch boundaries
eb = [steps[i] for i in range(1, len(epochs)) if epochs[i] != epochs[i - 1]]

ax1.set_xlabel("Outer Step")
ax1.set_ylabel("Loss", color="black")
l1, = ax1.plot(steps, lfgt, color="#d62728", linewidth=1.8, label="$\\mathcal{L}_{\\mathrm{forget}}$")
l2, = ax1.plot(steps, lret, color="#1f77b4", linewidth=1.5, label="$\\mathcal{L}_{\\mathrm{retain}}$")
ax1.axhline(0.15, color="#ff7f0e", linestyle="-.", linewidth=1, alpha=0.5)
ax1.set_ylim(-0.2, 8.0)
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
ax2.set_ylabel("$\\lambda$", color="#9467bd")
l3, = ax2.plot(steps, lam, color="#9467bd", linewidth=1.5, linestyle="--", label="$\\lambda$")
ax2.tick_params(axis="y", labelcolor="#9467bd")
ax2.set_ylim(0.5, 2.8)

for xb in eb:
    ax1.axvline(xb, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
           loc="upper right", framealpha=0.95)
ax1.set_title("MUSE Books — Llama-2-7B: Clamped Entropy + ALM, 4 Epochs (HM = 0.798)")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_muse_books_dynamics.png")
fig.savefig(OUT_DIR / "fig1_muse_books_dynamics.pdf")
plt.close(fig)
print("Figure 1 (MUSE Books) saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: TOFU 1B — combined loss + λ
# ═══════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 4.5))

steps_t = extract(h_tofu, "step")
lfgt_t = extract(h_tofu, "L_fgt")
lret_t = extract(h_tofu, "L_ret")
lam_t = extract(h_tofu, "lambda")
epochs_t = extract(h_tofu, "epoch")

eb_t = [steps_t[i] for i in range(1, len(epochs_t)) if epochs_t[i] != epochs_t[i - 1]]

ax1.set_xlabel("Outer Step")
ax1.set_ylabel("Loss", color="black")
l1, = ax1.plot(steps_t, lfgt_t, color="#d62728", linewidth=1.8, label="$\\mathcal{L}_{\\mathrm{forget}}$")
l2, = ax1.plot(steps_t, lret_t, color="#1f77b4", linewidth=1.5, label="$\\mathcal{L}_{\\mathrm{retain}}$")
ax1.axhline(0.15, color="#ff7f0e", linestyle="-.", linewidth=1, alpha=0.5, label="$\\varepsilon$")
ax1.set_ylim(-0.1, 8.0)
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
ax2.set_ylabel("$\\lambda$", color="#9467bd")
l3, = ax2.plot(steps_t, lam_t, color="#9467bd", linewidth=1.5, linestyle="--", label="$\\lambda$")
ax2.tick_params(axis="y", labelcolor="#9467bd")
ax2.set_ylim(0.8, 2.6)

for xb in eb_t:
    ax1.axvline(xb, color="gray", linestyle="--", alpha=0.25, linewidth=0.8)

ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
           loc="right", framealpha=0.95)
ax1.set_title("TOFU forget01 — Llama-3.2-1B: Clamped Entropy + ALM, 150 Steps")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_tofu_1b_dynamics.png")
fig.savefig(OUT_DIR / "fig2_tofu_1b_dynamics.pdf")
plt.close(fig)
print("Figure 2 (TOFU 1B) saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: TOFU 3B — combined loss + λ
# ═══════════════════════════════════════════════════════════════════════════
h_tofu3b = load_history(SAVES / "unlearn/tofu_Llama-3.2-3B-Instruct_forget01_LoRABiAL_emult0.85/lora_bial_history.json")

fig, ax1 = plt.subplots(figsize=(10, 4.5))

steps_3 = extract(h_tofu3b, "step")
lfgt_3 = extract(h_tofu3b, "L_fgt")
lret_3 = extract(h_tofu3b, "L_ret")
lam_3 = extract(h_tofu3b, "lambda")
epochs_3 = extract(h_tofu3b, "epoch")

eb_3 = [steps_3[i] for i in range(1, len(epochs_3)) if epochs_3[i] != epochs_3[i - 1]]

ax1.set_xlabel("Outer Step")
ax1.set_ylabel("Loss", color="black")
l1, = ax1.plot(steps_3, lfgt_3, color="#d62728", linewidth=1.8, label="$\\mathcal{L}_{\\mathrm{forget}}$")
l2, = ax1.plot(steps_3, lret_3, color="#1f77b4", linewidth=1.5, label="$\\mathcal{L}_{\\mathrm{retain}}$")

eps_3 = h_tofu3b[0]["r"] + h_tofu3b[0]["L_ret"]
ax1.axhline(eps_3, color="#ff7f0e", linestyle="-.", linewidth=1, alpha=0.5, label="$\\varepsilon$")
ax1.set_ylim(-0.1, 8.0)
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
ax2.set_ylabel("$\\lambda$", color="#9467bd")
l3, = ax2.plot(steps_3, lam_3, color="#9467bd", linewidth=1.5, linestyle="--", label="$\\lambda$")
ax2.tick_params(axis="y", labelcolor="#9467bd")
ax2.set_ylim(0.8, 2.8)

for xb in eb_3:
    ax1.axvline(xb, color="gray", linestyle="--", alpha=0.25, linewidth=0.8)

ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]],
           loc="right", framealpha=0.95)
ax1.set_title("TOFU forget01 — Llama-3.2-3B: Clamped Entropy + ALM, 200 Steps")

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_tofu_3b_dynamics.png")
fig.savefig(OUT_DIR / "fig3_tofu_3b_dynamics.pdf")
plt.close(fig)
print("Figure 3 (TOFU 3B) saved.")


print(f"\nAll figures saved to {OUT_DIR}/")
