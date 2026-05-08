"""
LoRA-BiAL-Adaptive: fully self-tuning LoRA-BiAL
================================================

Zero external constants. Every decision is derived from the ALM's own signals:

1. **Auto-ε** (inherited): ε = multiplier × baseline_L_ret at step 0.

2. **LR Calibration** (steps 0 to calibration window):
   Observe the L_fgt decay rate relative to L_fgt_init and whether the
   constraint residual r = L_ret − ε breaches zero. At the calibration
   boundary, adjust outer LR so that:
   - L_fgt would drop ~50% over 1/3 of the safety cap T (target pace)
   - If r went positive early, scale down to stay in Phase I longer

3. **Phase Transition Detection**:
   T_trans = first step where r exceeds ε (the retain budget itself)
   sustained for 3 steps. No arbitrary threshold — ε IS the threshold.

4. **Convergence Stopping**:
   After T_trans + 0.6 × T_trans steps (the "controlled descent" phase
   needs ~60% of the free-forgetting phase to converge):
   Stop when L_fgt velocity (EMA) < 1% of its own peak velocity.
   This is purely relative — no dataset-dependent absolute thresholds.

The only user input is the starting LR and T (safety cap). The trainer
auto-adjusts LR at one checkpoint and stops when forgetting is exhausted.
"""

import logging
import os
import json
import math
from typing import Optional

from trainer.unlearn.lora_bial import LoRABiAL

logger = logging.getLogger(__name__)


class LoRABiALAdaptive(LoRABiAL):

    def __init__(
        self,
        *args,
        # Only structural params — no dataset-dependent thresholds
        calibration_frac: float = 0.10,
        conv_patience: int = 8,
        ema_alpha: float = 0.15,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.calibration_frac = calibration_frac
        self.conv_patience = conv_patience
        self.ema_alpha = ema_alpha

    def _init_state(self):
        return {
            # Core signals
            "lfgt_init": None,
            "lfgt_ema": None,
            "prev_lfgt_ema": None,
            "vel_ema": 0.0,
            "peak_vel": 0.0,
            # LR calibration
            "lr_adjusted": False,
            "lr_scale": 1.0,
            "r_breached_early": False,
            "calibration_step": None,
            # T_trans
            "r_pos_streak": 0,
            "t_trans": None,
            "min_stop_step": None,
            # Convergence
            "conv_count": 0,
            "stopped": False,
            "stop_step": None,
        }

    def _update_ema(self, state, L_fgt):
        alpha = self.ema_alpha
        if state["lfgt_init"] is None:
            state["lfgt_init"] = L_fgt
            state["lfgt_ema"] = L_fgt
            state["prev_lfgt_ema"] = L_fgt
            return

        state["lfgt_ema"] = alpha * L_fgt + (1 - alpha) * state["lfgt_ema"]
        vel = state["lfgt_ema"] - state["prev_lfgt_ema"]
        state["vel_ema"] = alpha * vel + (1 - alpha) * state["vel_ema"]
        state["prev_lfgt_ema"] = state["lfgt_ema"]

        if abs(state["vel_ema"]) > state["peak_vel"]:
            state["peak_vel"] = abs(state["vel_ema"])

    def _calibrate_lr(self, step, L_fgt, r, state, max_steps):
        """One-shot LR adjustment at the calibration boundary.

        Target pace: L_fgt should drop ~50% over the first third of T.
        This means the observed per-step fractional decay should be
        approximately -ln(0.5) / (T/3) = 2.08/T.
        If it's faster → scale down. If slower → scale up.
        The ratio is bounded to [0.3, 3.0] to avoid wild swings.

        Safety override: if r went positive before calibration,
        LR is too high for this ε — scale down by 0.5.
        """
        cal_step = int(self.calibration_frac * max_steps)
        cal_step = max(5, cal_step)
        state["calibration_step"] = cal_step

        if state["lr_adjusted"] or step < cal_step:
            if r > self.epsilon and step < cal_step:
                state["r_breached_early"] = True
            return

        state["lr_adjusted"] = True
        frac_remaining = state["lfgt_ema"] / state["lfgt_init"]
        observed_rate = -math.log(max(frac_remaining, 0.01)) / step

        target_rate = -math.log(0.5) / (max_steps / 3.0)

        scale = 1.0
        reason = "on target"

        if state["r_breached_early"]:
            scale = 0.5
            reason = f"r breached ε={self.epsilon:.4f} during calibration"
        else:
            scale = target_rate / max(observed_rate, 1e-8)
            scale = max(0.3, min(3.0, scale))
            reason = (f"obs_rate={observed_rate:.5f}, target={target_rate:.5f}, "
                      f"L_fgt dropped to {frac_remaining:.1%}")

        if abs(scale - 1.0) > 0.08:
            old_lr = self._outer_opt.param_groups[0]["lr"]
            new_lr = old_lr * scale
            for pg in self._outer_opt.param_groups:
                pg["lr"] = new_lr
            state["lr_scale"] = scale
            logger.info(f"  LR-CALIBRATE @ step {step}: {old_lr:.2e} → {new_lr:.2e} "
                        f"(×{scale:.2f}, {reason})")
        else:
            logger.info(f"  LR-CALIBRATE @ step {step}: no adjustment ({reason})")

    def _detect_transition(self, step, r, state):
        """T_trans = first step where r > ε sustained for 3 steps.

        Using ε as the threshold is natural: when L_ret overshoots ε by ε,
        the constraint is definitely active. No arbitrary threshold needed.
        """
        if state["t_trans"] is not None:
            return

        if r > self.epsilon:
            state["r_pos_streak"] += 1
            if state["r_pos_streak"] >= 3:
                state["t_trans"] = step - 2
                state["min_stop_step"] = int(state["t_trans"] * 1.6)
                logger.info(f"  PHASE-TRANS @ step {step}: T_trans={state['t_trans']}, "
                            f"min_stop={state['min_stop_step']}")
        else:
            state["r_pos_streak"] = 0

    def _check_convergence(self, step, state):
        """Stop when L_fgt velocity drops below 1% of its peak velocity.

        Purely relative — the peak velocity is measured from Phase I/II,
        and we wait for the signal to die to 1% of that peak. This
        adapts automatically to any dataset/LR combination.

        Prerequisites:
          - Past T_trans + 60% of T_trans (Phase III needs time)
          - L_fgt has dropped at least 50% from init
          - Peak velocity was actually observed (not step 0)
        """
        if state["t_trans"] is None:
            return False
        if state["min_stop_step"] is not None and step < state["min_stop_step"]:
            return False
        if state["peak_vel"] < 1e-8:
            return False

        drop_frac = 1.0 - (state["lfgt_ema"] / state["lfgt_init"])
        if drop_frac < 0.50:
            return False

        # Two convergence signals (either suffices):
        # 1. Velocity dropped to <1% of peak (standard)
        # 2. L_fgt_ema dropped to <2% of init (effectively zero)
        vel_ratio = abs(state["vel_ema"]) / state["peak_vel"]
        lfgt_ratio = state["lfgt_ema"] / state["lfgt_init"]
        if vel_ratio < 0.01 or lfgt_ratio < 0.02:
            state["conv_count"] += 1
            if state["conv_count"] >= self.conv_patience:
                state["stopped"] = True
                state["stop_step"] = step - self.conv_patience + 1
                return True
        else:
            state["conv_count"] = 0

        return False

    def train(self):
        import torch
        import time
        from torch.utils.data import DataLoader

        device = self.args.device

        logger.info("=" * 60)
        logger.info("LoRA-BiAL Adaptive (self-tuning LR + convergence)")
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
        steps_per_epoch = max(1, forget_micro_batches // self.gradient_accumulation_steps)
        num_epochs = max(1, int(self.args.num_train_epochs))
        max_outer_steps = self.T if self.T > 0 else num_epochs * steps_per_epoch

        logger.info(f"  Safety cap T={max_outer_steps}, eff_bs={effective_bs}, K={self.K}")
        logger.info(f"  Forget loss: {self.forget_loss_type}")
        logger.info(f"  ALM: ε_mul={self.epsilon_multiplier}, ρ={self.rho}, λ_init={self.lambda_init}")
        logger.info(f"  Start LR: outer={self.eta_theta}, inner={self.eta_in}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        logger.info(f"  Adaptive: cal_frac={self.calibration_frac}, "
                     f"conv_patience={self.conv_patience}, ema_α={self.ema_alpha}")

        history = []
        log_every = max(1, max_outer_steps // 30)
        state = self._init_state()
        resume_step = 0

        # Check for resumable checkpoint
        if self.checkpoint_every_n_steps > 0:
            import glob
            ckpt_dirs = sorted(glob.glob(os.path.join(self.args.output_dir, "checkpoint-step*")))
            if ckpt_dirs:
                latest_ckpt = ckpt_dirs[-1]
                resume_step = int(latest_ckpt.split("checkpoint-step")[-1])
                logger.info(f"  Resuming from {latest_ckpt} (step {resume_step})")
                from peft import PeftModel
                if isinstance(self.model, PeftModel):
                    self.model.load_adapter(latest_ckpt, adapter_name="default")
                else:
                    self.model = PeftModel.from_pretrained(self.model, latest_ckpt)
                self.model.train()

        # Auto-epsilon
        inner_losses = self.inner_loop(device)
        if inner_losses:
            baseline_ret = sum(inner_losses) / len(inner_losses)
        else:
            # K=0: estimate baseline retain loss without optimizer step
            with torch.no_grad():
                batch = self._next_retain_batch()
                baseline_ret = self._compute_ce_loss(batch, device).item()
        self.epsilon = self.epsilon_multiplier * baseline_ret
        logger.info(f"  Auto-ε: inner_avg={baseline_ret:.4f}, "
                    f"multiplier={self.epsilon_multiplier}, ε={self.epsilon:.4f}")

        for t in range(resume_step, max_outer_steps):
            t_start = time.time()
            epoch = t // steps_per_epoch if steps_per_epoch > 0 else 0

            # Inner loop (skip step 0 — auto-ε already ran it)
            if t == 0:
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

            dt = time.time() - t_start
            inner_mean = sum(inner_losses) / len(inner_losses) if inner_losses else 0.0

            # --- Adaptive signals ---
            self._update_ema(state, L_fgt)
            self._calibrate_lr(t, L_fgt, r, state, max_outer_steps)
            self._detect_transition(t, r, state)
            converged = self._check_convergence(t, state)

            hist_entry = {
                "step": t, "epoch": epoch, "L_fgt": L_fgt, "L_ret": L_ret,
                "r": r, "lambda": self.lambda_dual, "inner_loss_mean": inner_mean,
                "dt": dt, "lr": self._outer_opt.param_groups[0]["lr"],
                "lfgt_ema": state["lfgt_ema"] if state["lfgt_ema"] is not None else L_fgt,
                "vel_ema": state["vel_ema"],
                "peak_vel": state["peak_vel"],
                "t_trans": state["t_trans"],
                "converged": state["stopped"],
            }
            history.append(hist_entry)

            if t % log_every == 0 or t == max_outer_steps - 1 or converged:
                extras = f" lr={self._outer_opt.param_groups[0]['lr']:.2e}"
                if state["t_trans"] is not None:
                    extras += f" T_tr={state['t_trans']}"
                if state["conv_count"] > 0:
                    extras += f" conv={state['conv_count']}/{self.conv_patience}"
                vel_pct = (abs(state["vel_ema"]) / state["peak_vel"] * 100
                           if state["peak_vel"] > 0 else 0)
                extras += f" v%={vel_pct:.1f}"
                logger.info(
                    f"  [{t:4d}/{max_outer_steps}|e{epoch+1}] "
                    f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                    f"r={r:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner={inner_mean:.4f}{extras} dt={dt:.1f}s"
                )

            if converged:
                logger.info(f"  CONVERGED at step {t+1} "
                            f"(T_trans={state['t_trans']}, min_stop={state['min_stop_step']}, "
                            f"vel={state['vel_ema']:.6f}, peak_vel={state['peak_vel']:.6f}, "
                            f"vel_ratio={abs(state['vel_ema'])/state['peak_vel']:.4f})")
                ckpt_dir = os.path.join(self.args.output_dir, "converged-best")
                self._save_checkpoint(ckpt_dir, history)
                logger.info(f"  Convergence checkpoint saved to {ckpt_dir}")
                break

            global_step = t + 1

            if global_step in self.eval_at_steps:
                ckpt_dir = os.path.join(self.args.output_dir, f"step-{global_step}")
                logger.info(f"  Saving intermediate checkpoint at step {global_step}...")
                self._save_checkpoint(ckpt_dir, history)

            if (self.checkpoint_every_n_steps > 0
                    and global_step % self.checkpoint_every_n_steps == 0
                    and global_step < max_outer_steps):
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-step{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                logger.info(f"  LoRA checkpoint saved: {ckpt_dir}")

            if (self.checkpoint_every_epoch and steps_per_epoch > 0
                    and global_step % steps_per_epoch == 0
                    and global_step < max_outer_steps):
                ep_num = global_step // steps_per_epoch
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{ep_num}")
                self._save_checkpoint(ckpt_dir, history)
                logger.info(f"  Epoch {ep_num} checkpoint saved.")

            if L_ret > 10.0:
                logger.warning(f"  L_ret={L_ret:.1f} > 10.0 — collapsed at step {global_step}.")
                break

        # Save LoRA adapters separately (for sequential chaining)
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.save_lora_only:
            lora_dir = os.path.join(output_dir, "lora_adapters")
            os.makedirs(lora_dir, exist_ok=True)
            self.model.save_pretrained(lora_dir)
            logger.info(f"LoRA adapters saved to {lora_dir}")

        # Final merge and save
        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "lora_bial_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Model saved to {output_dir}")
        if history:
            last = history[-1]
            logger.info(f"Final step {last['step']+1}: λ={self.lambda_dual:.3f}, "
                         f"L_fgt={last['L_fgt']:.4f}, L_ret={last['L_ret']:.4f}, "
                         f"lr={last['lr']:.2e}")
        if state["stopped"]:
            logger.info(f"Adaptive: converged at step {state['stop_step']+1}, "
                         f"T_trans={state['t_trans']}, lr_scale={state['lr_scale']:.2f}")
        else:
            logger.info(f"Adaptive: hit safety cap T={max_outer_steps} without convergence")

        self.evaluate()
