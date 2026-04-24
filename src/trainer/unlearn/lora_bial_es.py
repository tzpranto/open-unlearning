"""
LoRA-BiAL-ES: LoRA-BiAL with automatic Early Stopping
======================================================

Extends LoRA-BiAL with a triple-conjunction criterion that detects the end of
the forgetting transition phase and stops training automatically.

The criterion monitors three EMA-smoothed signals:
  C1 (λ deceleration): dλ/dt decayed past 50% of its peak
  C2 (sufficient forgetting): L_fgt dropped by ≥75% from initial
  C3 (velocity saturation): |dL_fgt/dt| < 1.4% of L_fgt_init

All three must hold for `patience` consecutive steps. When the criterion fires,
a "best" checkpoint is saved and training continues for `margin` more steps
before hard-stopping. T remains as an absolute safety cap.

Validated against known sweet spots:
  TOFU 1B:      fires step 99  (sweet spot T=100, δ=-1)
  TOFU 3B:      fires step 79  (sweet spot T=75,  δ=+4)
  MUSE Books 7B: fires step 59 (sweet spot T=60,  δ=-1)
"""

import logging
import os
import json
from typing import Optional

from trainer.unlearn.lora_bial import LoRABiAL

logger = logging.getLogger(__name__)


class LoRABiALES(LoRABiAL):

    def __init__(
        self,
        *args,
        # Auto-stop parameters
        auto_stop: bool = True,
        auto_stop_alpha: float = 0.20,
        auto_stop_patience: int = 3,
        auto_stop_margin: int = 10,
        auto_stop_dlam_decay: float = 0.5,
        auto_stop_lfgt_drop: float = 0.75,
        auto_stop_dfgt_rel: float = 0.014,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.auto_stop = auto_stop
        self.auto_stop_alpha = auto_stop_alpha
        self.auto_stop_patience = auto_stop_patience
        self.auto_stop_margin = auto_stop_margin
        self.auto_stop_dlam_decay = auto_stop_dlam_decay
        self.auto_stop_lfgt_drop = auto_stop_lfgt_drop
        self.auto_stop_dfgt_rel = auto_stop_dfgt_rel

    def _init_auto_stop(self):
        self._as = {
            "lfgt_ema": None, "dlam_ema": 0.0, "dfgt_ema": 0.0,
            "prev_lam": None, "prev_lfgt_ema": None,
            "peak_dlam": 0.0, "transition_entered": False,
            "consec": 0, "candidate_step": None,
            "fired_step": None, "lfgt_init": None,
        }

    def _check_auto_stop(self, step, L_fgt, lam):
        s = self._as
        alpha = self.auto_stop_alpha

        if s["lfgt_ema"] is None:
            s["lfgt_ema"] = L_fgt
            s["prev_lam"] = lam
            s["prev_lfgt_ema"] = L_fgt
            s["lfgt_init"] = L_fgt
            return False, {}

        s["lfgt_ema"] = alpha * L_fgt + (1 - alpha) * s["lfgt_ema"]

        dlam = alpha * (lam - s["prev_lam"]) + (1 - alpha) * s["dlam_ema"]
        s["dlam_ema"] = dlam

        dfgt = alpha * (s["lfgt_ema"] - s["prev_lfgt_ema"]) + (1 - alpha) * s["dfgt_ema"]
        s["dfgt_ema"] = dfgt

        s["prev_lam"] = lam
        s["prev_lfgt_ema"] = s["lfgt_ema"]

        if dlam > 0.003:
            s["transition_entered"] = True
        if dlam > s["peak_dlam"]:
            s["peak_dlam"] = dlam
            s["consec"] = 0
            s["candidate_step"] = None

        c1 = (s["transition_entered"]
              and s["peak_dlam"] > 0.003
              and dlam < s["peak_dlam"] * self.auto_stop_dlam_decay)
        c2 = s["lfgt_ema"] < s["lfgt_init"] * (1.0 - self.auto_stop_lfgt_drop)
        c3 = abs(dfgt) < self.auto_stop_dfgt_rel * s["lfgt_init"]

        if c1 and c2 and c3:
            s["consec"] += 1
            if s["consec"] == 1:
                s["candidate_step"] = step
            if s["consec"] >= self.auto_stop_patience and s["fired_step"] is None:
                s["fired_step"] = s["candidate_step"]
                return True, {
                    "fired_step": s["fired_step"],
                    "peak_dlam": s["peak_dlam"],
                    "lfgt_ema": s["lfgt_ema"],
                    "dlam_ema": dlam,
                    "dfgt_ema": dfgt,
                }
        else:
            s["consec"] = 0
            s["candidate_step"] = None

        return False, {}

    def train(self):
        import torch
        import time
        from torch.utils.data import DataLoader

        device = self.args.device

        # LoRA adapters
        logger.info("=" * 60)
        logger.info("LoRA adapters (with Early Stopping)")
        logger.info("=" * 60)
        self._wrap_with_lora()

        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.SGD(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)
        logger.info(f"Optimizers: inner=SGD(lr={self.eta_in}), outer=Adam(lr={self.eta_theta}), "
                     f"params={sum(p.numel() for p in lora_params)}")

        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Data
        forget_ds = self.train_dataset.forget
        retain_ds = self.train_dataset.retain
        collator = self.data_collator
        self._forget_dataloader = DataLoader(
            forget_ds, batch_size=self.args.per_device_train_batch_size,
            shuffle=True, collate_fn=collator, drop_last=False, pin_memory=True,
        )
        self._retain_dataloader = DataLoader(
            retain_ds, batch_size=self.args.per_device_train_batch_size,
            shuffle=True, collate_fn=collator, drop_last=False, pin_memory=True,
        )
        self._forget_iter = iter(self._forget_dataloader)
        self._retain_iter = iter(self._retain_dataloader)

        batch_size = self.args.per_device_train_batch_size
        effective_bs = batch_size * self.gradient_accumulation_steps
        inner_effective_bs = batch_size * self.inner_accumulation_steps
        forget_micro_batches = len(self._forget_dataloader)
        retain_micro_batches = len(self._retain_dataloader)
        steps_per_epoch = max(1, forget_micro_batches // self.gradient_accumulation_steps)
        num_epochs = max(1, int(self.args.num_train_epochs))
        max_outer_steps = self.T if self.T > 0 else num_epochs * steps_per_epoch

        # LR schedulers
        self._outer_scheduler = None
        self._inner_scheduler = None
        if self.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
            warmup_steps = int(self.warmup_fraction * max_outer_steps)
            if warmup_steps > 0:
                self._outer_scheduler = SequentialLR(self._outer_opt, [
                    LinearLR(self._outer_opt, start_factor=0.1, total_iters=warmup_steps),
                    CosineAnnealingLR(self._outer_opt, T_max=max_outer_steps - warmup_steps),
                ], milestones=[warmup_steps])
                self._inner_scheduler = SequentialLR(self._inner_opt, [
                    LinearLR(self._inner_opt, start_factor=0.1, total_iters=warmup_steps),
                    CosineAnnealingLR(self._inner_opt, T_max=max_outer_steps - warmup_steps),
                ], milestones=[warmup_steps])
            else:
                self._outer_scheduler = CosineAnnealingLR(self._outer_opt, T_max=max_outer_steps)
                self._inner_scheduler = CosineAnnealingLR(self._inner_opt, T_max=max_outer_steps)

        # Log config
        logger.info("=" * 60)
        logger.info("Bilevel ALM optimization")
        logger.info("=" * 60)
        logger.info(f"  Mode: {'epoch-based' if self.T <= 0 else f'fixed T={self.T}'}")
        logger.info(f"  Epochs={num_epochs}, micro_bs={batch_size}, "
                     f"outer_accum={self.gradient_accumulation_steps} (eff_bs={effective_bs}), "
                     f"inner_accum={self.inner_accumulation_steps} (eff_bs={inner_effective_bs})")
        logger.info(f"  forget_batches/epoch={forget_micro_batches}, "
                     f"retain_batches/epoch={retain_micro_batches}, "
                     f"outer_steps/epoch={steps_per_epoch}, "
                     f"max_steps={max_outer_steps}, K={self.K}")
        logger.info(f"  Forget loss: {self.forget_loss_type} (beta={self.npo_beta})")
        eps_str = f"ε_mul={self.epsilon_multiplier}" if self.epsilon_multiplier > 0 else f"ε={self.epsilon}"
        logger.info(f"  ALM: {eps_str}, ρ={self.rho}, λ_init={self.lambda_init}"
                     f"{f', λ_max={self.lambda_max}' if self.lambda_max > 0 else ''}")
        logger.info(f"  LR: outer={self.eta_theta}, inner={self.eta_in}, schedule={self.lr_schedule}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        if self.auto_stop:
            logger.info(f"  Auto-stop: α={self.auto_stop_alpha}, patience={self.auto_stop_patience}, "
                         f"margin={self.auto_stop_margin}, dlam_decay={self.auto_stop_dlam_decay}, "
                         f"lfgt_drop={self.auto_stop_lfgt_drop}, dfgt_rel={self.auto_stop_dfgt_rel}")
        if self.use_pcgrad:
            logger.info("  PCGrad: enabled")
        if self.inner_warmup_steps > 0:
            logger.info(f"  Inner warmup: outer-only for first {self.inner_warmup_steps} steps")
        if self.use_implicit:
            logger.info(f"  Implicit: FD-HVP Neumann, steps={self.neumann_steps}, "
                         f"μ={self.neumann_mu}, eps={self.fd_hvp_eps}, warmup={self.implicit_warmup_steps}")
        if torch.cuda.is_available():
            logger.info(f"  GPU memory before bilevel: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # Training loop
        history = []
        log_every = max(1, max_outer_steps // 20)

        # Auto-epsilon
        if self.epsilon_multiplier > 0:
            inner_losses = self.inner_loop(device)
            baseline_ret = sum(inner_losses) / len(inner_losses)
            self.epsilon = self.epsilon_multiplier * baseline_ret
            logger.info(f"  Auto-ε: inner_avg={baseline_ret:.4f}, "
                        f"multiplier={self.epsilon_multiplier}, ε={self.epsilon:.4f}")

        # Auto-stop state
        if self.auto_stop:
            self._init_auto_stop()
            as_ckpt_saved = False
            as_hard_deadline = None

        for t in range(max_outer_steps):
            t_start = time.time()
            epoch = t // steps_per_epoch if steps_per_epoch > 0 else 0

            # Inner loop
            if t == 0 and self.epsilon_multiplier > 0:
                pass
            elif t < self.inner_warmup_steps:
                inner_losses = []
            else:
                inner_losses = self.inner_loop(device)

            # Outer step
            L_fgt, L_ret, r = self.outer_step(device, global_step=t)

            # Adaptive inner recovery
            extra_inner = 0
            while L_ret > 2 * self.epsilon and extra_inner < self.K * 3:
                self.inner_loop(device)
                with torch.no_grad():
                    rb = self._next_retain_batch()
                    L_ret = self._compute_ce_loss(rb, device).item()
                r = L_ret - self.epsilon
                extra_inner += self.K
            if extra_inner > 0:
                logger.info(f"  Adaptive inner: {extra_inner} extra steps, L_ret={L_ret:.4f}")

            # LR schedulers
            if self._outer_scheduler is not None:
                self._outer_scheduler.step()
                if t >= self.inner_warmup_steps and self._inner_scheduler is not None:
                    self._inner_scheduler.step()

            dt = time.time() - t_start
            inner_mean = sum(inner_losses) / len(inner_losses) if inner_losses else 0.0
            hist_entry = {
                "step": t, "epoch": epoch, "L_fgt": L_fgt, "L_ret": L_ret,
                "r": r, "lambda": self.lambda_dual, "inner_loss_mean": inner_mean,
                "dt": dt,
            }

            # Auto-stop check
            if self.auto_stop:
                fired, info = self._check_auto_stop(t, L_fgt, self.lambda_dual)
                hist_entry["dlam_ema"] = self._as["dlam_ema"]
                hist_entry["dfgt_ema"] = self._as["dfgt_ema"]
                hist_entry["lfgt_ema"] = self._as["lfgt_ema"] if self._as["lfgt_ema"] is not None else L_fgt
                hist_entry["auto_stop_fired"] = self._as["fired_step"] is not None

                if fired and not as_ckpt_saved:
                    ckpt_dir = os.path.join(self.args.output_dir, "auto-stop-best")
                    logger.info(f"  AUTO-STOP: criterion fired at step {info['fired_step']}! "
                                 f"Saving best checkpoint...")
                    logger.info(f"    peak_dlam={info['peak_dlam']:.5f}, "
                                 f"lfgt_ema={info['lfgt_ema']:.4f}, "
                                 f"dlam_ema={info['dlam_ema']:.5f}, "
                                 f"dfgt_ema={info['dfgt_ema']:.5f}")
                    history.append(hist_entry)
                    self._save_checkpoint(ckpt_dir, history)
                    as_ckpt_saved = True
                    as_hard_deadline = t + self.auto_stop_margin
                    logger.info(f"    Will hard-stop at step {as_hard_deadline} "
                                 f"(margin={self.auto_stop_margin})")
                    # Skip appending again
                    hist_entry = None

                if as_hard_deadline is not None and t >= as_hard_deadline:
                    if hist_entry is not None:
                        history.append(hist_entry)
                    logger.info(f"  AUTO-STOP: hard stopping at step {t+1} "
                                 f"(best checkpoint at step {self._as['fired_step']+1}).")
                    break

            if hist_entry is not None:
                history.append(hist_entry)

            # Logging
            if t % log_every == 0 or t == max_outer_steps - 1:
                extras = ""
                if self._outer_scheduler is not None:
                    extras += f" olr={self._outer_opt.param_groups[0]['lr']:.2e}"
                if self.use_pcgrad and hasattr(self, '_pcgrad_cos'):
                    extras += f" cos={self._pcgrad_cos:+.3f}"
                if self.auto_stop and self._as["fired_step"] is not None:
                    extras += f" AS@{self._as['fired_step']+1}"
                logger.info(
                    f"  [{t:4d}/{max_outer_steps}|e{epoch+1}] "
                    f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                    f"r={r:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner={inner_mean:.4f}{extras} dt={dt:.1f}s"
                )

            global_step = t + 1

            # Intermediate step checkpoint
            if global_step in self.eval_at_steps:
                ckpt_dir = os.path.join(self.args.output_dir, f"step-{global_step}")
                logger.info(f"  Saving intermediate checkpoint at step {global_step}...")
                self._save_checkpoint(ckpt_dir, history)
                logger.info(f"  Checkpoint step-{global_step} saved.")

            # Per-epoch checkpoint
            if (self.checkpoint_every_epoch and steps_per_epoch > 0
                    and global_step % steps_per_epoch == 0
                    and global_step < max_outer_steps):
                ep_num = global_step // steps_per_epoch
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{ep_num}")
                self._save_checkpoint(ckpt_dir, history)
                logger.info(f"  Epoch {ep_num} checkpoint saved: {ckpt_dir}")

            if L_ret > 10.0:
                logger.warning(f"  L_ret={L_ret:.1f} > 10.0 — model collapsed at step {global_step}.")
                break

        # Merge LoRA and save final model
        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "lora_bial_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Model saved to {output_dir}")
        if history:
            logger.info(f"Final: λ={self.lambda_dual:.3f}, "
                         f"L_fgt={history[-1]['L_fgt']:.4f}, L_ret={history[-1]['L_ret']:.4f}")
        if self.auto_stop and self._as["fired_step"] is not None:
            logger.info(f"Auto-stop best checkpoint: step {self._as['fired_step']+1} "
                         f"(saved to auto-stop-best/)")

        self.evaluate()
