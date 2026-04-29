"""
BiAL-Full-Adaptive: Full fine-tune version of LoRA-BiAL-Adaptive
================================================================

Same bilevel ALM optimization as LoRA-BiAL-Adaptive but without LoRA —
all model parameters are trained directly. Used for ablation to prove
LoRA is a meaningful architectural choice (parameter efficiency +
implicit regularization via low-rank constraint).

Memory-heavy: requires gradient_checkpointing and lower batch sizes.
"""

import logging
import os
import json
import time
import torch
from torch.utils.data import DataLoader

from trainer.unlearn.lora_bial_adaptive import LoRABiALAdaptive

logger = logging.getLogger(__name__)


class BiALFullAdaptive(LoRABiALAdaptive):

    def _wrap_with_lora(self):
        """No-op: skip LoRA wrapping, train all params directly."""
        for param in self.model.parameters():
            param.requires_grad = True
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Full fine-tune mode: all {n_params:,} parameters trainable")

    def inner_loop(self, device):
        """Inner loop without adapter layer calls."""
        self.model.train()
        inner_losses = []
        for k in range(self.K):
            self._inner_opt.zero_grad()
            accum_loss = 0.0
            for _ in range(self.inner_accumulation_steps):
                batch = self._next_retain_batch()
                loss = self._compute_ce_loss(batch, device)
                (loss / self.inner_accumulation_steps).backward()
                accum_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self._inner_opt.step()
            inner_losses.append(accum_loss / self.inner_accumulation_steps)
        return inner_losses

    def outer_step(self, device, global_step=0):
        """Outer step without adapter layer calls."""
        self.model.train()
        self._outer_opt.zero_grad()

        total_L_fgt = 0.0
        total_L_ret = 0.0

        for _ in range(self.gradient_accumulation_steps):
            forget_batch = self._next_forget_batch()
            retain_batch = self._next_retain_batch()

            L_fgt = self._compute_forget_loss(forget_batch, device)
            L_ret = self._compute_ce_loss(retain_batch, device)

            r_micro = L_ret - self.epsilon
            r_plus = torch.clamp(r_micro, min=0.0)
            L_alm = L_fgt + self.lambda_dual * r_micro + 0.5 * self.rho * (r_plus ** 2)
            (L_alm / self.gradient_accumulation_steps).backward()

            total_L_fgt += L_fgt.item()
            total_L_ret += L_ret.item()

        avg_L_fgt = total_L_fgt / self.gradient_accumulation_steps
        avg_L_ret = total_L_ret / self.gradient_accumulation_steps
        avg_r = avg_L_ret - self.epsilon

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        self._outer_opt.step()

        # Asymmetric dual update
        if avg_r > 0:
            self.lambda_dual += self.rho * avg_r
        else:
            self.lambda_dual += self.dual_decay_factor * self.rho * avg_r
        self.lambda_dual = max(self.lambda_min, self.lambda_dual)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return avg_L_fgt, avg_L_ret, avg_r

    def _save_checkpoint(self, ckpt_dir, history):
        """Save full model directly (no LoRA merge needed)."""
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.eval()
        self.model.save_pretrained(ckpt_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)
        self.model.train()
        with open(os.path.join(ckpt_dir, "lora_bial_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    def train(self):
        device = self.args.device

        logger.info("=" * 60)
        logger.info("BiAL-Full Adaptive (full fine-tune, no LoRA)")
        logger.info("=" * 60)
        self._wrap_with_lora()  # actually just enables all params

        all_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.SGD(all_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(all_params, lr=self.eta_theta)
        logger.info(f"Optimizers: inner=SGD(lr={self.eta_in}), outer=Adam(lr={self.eta_theta}), "
                     f"params={sum(p.numel() for p in all_params):,}")

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
        forget_micro_batches = len(self._forget_dataloader)
        steps_per_epoch = max(1, forget_micro_batches // self.gradient_accumulation_steps)
        num_epochs = max(1, int(self.args.num_train_epochs))
        max_outer_steps = self.T if self.T > 0 else num_epochs * steps_per_epoch

        logger.info(f"  Safety cap T={max_outer_steps}, eff_bs={effective_bs}, K={self.K}")
        logger.info(f"  Forget loss: {self.forget_loss_type}")
        logger.info(f"  ALM: ε_mul={self.epsilon_multiplier}, ρ={self.rho}, λ_init={self.lambda_init}")
        logger.info(f"  Start LR: outer={self.eta_theta}, inner={self.eta_in}")
        logger.info(f"  Full fine-tune (NO LoRA)")
        logger.info(f"  Adaptive: cal_frac={self.calibration_frac}, "
                     f"conv_patience={self.conv_patience}, ema_α={self.ema_alpha}")

        history = []
        log_every = max(1, max_outer_steps // 30)
        state = self._init_state()

        # Auto-epsilon
        inner_losses = self.inner_loop(device)
        if inner_losses:
            baseline_ret = sum(inner_losses) / len(inner_losses)
        else:
            with torch.no_grad():
                batch = self._next_retain_batch()
                baseline_ret = self._compute_ce_loss(batch, device).item()
        self.epsilon = self.epsilon_multiplier * baseline_ret
        logger.info(f"  Auto-ε: inner_avg={baseline_ret:.4f}, "
                    f"multiplier={self.epsilon_multiplier}, ε={self.epsilon:.4f}")

        for t in range(max_outer_steps):
            t_start = time.time()
            epoch = t // steps_per_epoch if steps_per_epoch > 0 else 0

            # Inner loop
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

            # Adaptive signals
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

            if L_ret > 10.0:
                logger.warning(f"  L_ret={L_ret:.1f} > 10.0 — collapsed at step {global_step}.")
                break

        # Save final model
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
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
