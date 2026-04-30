"""
LoRA-BiAL Swapped: Inner=Forget, Outer=Retain+ALM(forget constraint)
=====================================================================

Ablation variant where inner/outer roles are swapped:
  Inner loop (K SGD steps): minimize forget loss (push toward forgetting)
  Outer loop (1 Adam step): minimize retain CE, subject to constraint that
    forget loss stays HIGH (above threshold)
  ALM: L_ret + λ·(ε_fgt - L_fgt) + ρ/2·max(0, ε_fgt - L_fgt)²

This tests whether the bilevel structure (forget=outer, retain=inner) matters,
or if any bilevel formulation with ALM would work.
"""

import torch
import time
import logging
import os
import json
from typing import Optional

from torch.utils.data import DataLoader
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.lora_bial_losses import compute_ce_loss, FORGET_LOSS_DISPATCH

logger = logging.getLogger(__name__)


class LoRABiALSwapped(UnlearnTrainer):

    def __init__(
        self,
        *args,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        T: int = 250,
        K: int = 3,
        eta_theta: float = 3e-5,
        eta_in: float = 2e-4,
        epsilon_multiplier: float = 3.2,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 0.0,
        lambda_min: float = 0.1,
        dual_decay_factor: float = 0.1,
        forget_loss_type: str = "clamped_entropy",
        npo_beta: float = 4.0,
        clamped_entropy_tau: float = 0.7,
        checkpoint_every_epoch: bool = False,
        gradient_accumulation_steps: int = 8,
        inner_accumulation_steps: int = 0,
        max_grad_norm: float = 1.0,
        # Accept but ignore params from base LoRABiAL config
        lr_schedule: str = "constant",
        warmup_fraction: float = 0.0,
        eval_at_steps: Optional[list] = None,
        use_implicit: bool = False,
        use_pcgrad: bool = False,
        inner_warmup_steps: int = 0,
        conv_patience: int = 20,
        calibration_frac: float = 0.1,
        ema_alpha: float = 0.15,
        **kwargs,
    ):
        # Filter out unknown kwargs before passing to super
        known_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(('neumann_', 'fd_', 'implicit_', 'ga_', 'focal_', 'npo_saturation', 'saturation_', 'retain_only'))}
        super().__init__(*args, **known_kwargs)
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        self.epsilon_multiplier = epsilon_multiplier
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.dual_decay_factor = dual_decay_factor
        self.lambda_dual = float(lambda_init)
        self.forget_loss_type = forget_loss_type
        if forget_loss_type not in FORGET_LOSS_DISPATCH:
            raise ValueError(f"Unknown forget_loss_type: {forget_loss_type}. "
                             f"Available: {list(FORGET_LOSS_DISPATCH.keys())}")
        self.npo_beta = npo_beta
        self.clamped_entropy_tau = clamped_entropy_tau
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.inner_accumulation_steps = inner_accumulation_steps if inner_accumulation_steps > 0 else gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

    def _wrap_with_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(
            r=self.lora_r, lora_alpha=self.lora_alpha_val,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        logger.info(f"LoRA applied: r={self.lora_r}, alpha={self.lora_alpha_val}, "
                     f"modules={self.lora_target_modules}")

    def _compute_ce_loss(self, batch, device):
        return compute_ce_loss(self.model, batch, device)

    def _compute_forget_loss(self, batch, device):
        return FORGET_LOSS_DISPATCH[self.forget_loss_type](
            self.model, batch, device,
            npo_beta=self.npo_beta,
            clamped_entropy_tau=self.clamped_entropy_tau,
        )

    def _next_retain_batch(self):
        try:
            return next(self._retain_iter)
        except StopIteration:
            self._retain_iter = iter(self._retain_dataloader)
            return next(self._retain_iter)

    def _next_forget_batch(self):
        try:
            return next(self._forget_iter)
        except StopIteration:
            self._forget_iter = iter(self._forget_dataloader)
            return next(self._forget_iter)

    # ------------------------------------------------------------------
    # SWAPPED Inner loop: minimize FORGET loss (push toward forgetting)
    # ------------------------------------------------------------------
    def inner_loop(self, device):
        self.model.enable_adapter_layers()
        self.model.train()
        inner_losses = []
        for k in range(self.K):
            self._inner_opt.zero_grad()
            accum_loss = 0.0
            for _ in range(self.inner_accumulation_steps):
                batch = self._next_forget_batch()
                loss = self._compute_forget_loss(batch, device)
                (loss / self.inner_accumulation_steps).backward()
                accum_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self._inner_opt.step()
            inner_losses.append(accum_loss / self.inner_accumulation_steps)
        return inner_losses

    # ------------------------------------------------------------------
    # SWAPPED Outer step: minimize retain + ALM(forget must stay high)
    # Constraint: ε_fgt - L_fgt <= 0 (i.e., L_fgt >= ε_fgt)
    # ------------------------------------------------------------------
    def outer_step(self, device):
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        total_L_fgt = 0.0
        total_L_ret = 0.0

        for _ in range(self.gradient_accumulation_steps):
            forget_batch = self._next_forget_batch()
            retain_batch = self._next_retain_batch()

            L_fgt = self._compute_forget_loss(forget_batch, device)
            L_ret = self._compute_ce_loss(retain_batch, device)

            # Swapped constraint: forget must stay HIGH (above epsilon)
            # r = epsilon_fgt - L_fgt (positive when forget is too low = bad)
            r_micro = self.epsilon_fgt - L_fgt
            r_plus = torch.clamp(r_micro, min=0.0)
            L_outer = L_ret + self.lambda_dual * r_micro + 0.5 * self.rho * (r_plus ** 2)
            (L_outer / self.gradient_accumulation_steps).backward()

            total_L_fgt += L_fgt.item()
            total_L_ret += L_ret.item()

        avg_L_fgt = total_L_fgt / self.gradient_accumulation_steps
        avg_L_ret = total_L_ret / self.gradient_accumulation_steps
        avg_r = self.epsilon_fgt - avg_L_fgt

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        self._outer_opt.step()

        # Dual update (same asymmetric logic)
        if avg_r > 0:
            self.lambda_dual += self.rho * avg_r
        else:
            self.lambda_dual += self.dual_decay_factor * self.rho * avg_r
        self.lambda_dual = max(self.lambda_min, self.lambda_dual)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return avg_L_fgt, avg_L_ret, avg_r

    def train(self):
        device = self.args.device

        logger.info("=" * 60)
        logger.info("LoRA-BiAL SWAPPED (inner=forget, outer=retain+ALM)")
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

        effective_bs = self.args.per_device_train_batch_size * self.gradient_accumulation_steps
        max_outer_steps = self.T

        logger.info(f"  T={self.T}, K={self.K}, eff_bs={effective_bs}")
        logger.info(f"  Forget loss (INNER): {self.forget_loss_type}")
        logger.info(f"  ALM constraint: L_fgt >= ε_fgt (penalize if forget drops)")
        logger.info(f"  ALM: ε_mul={self.epsilon_multiplier}, ρ={self.rho}, λ_init={self.lambda_init}")
        logger.info(f"  LR: outer={self.eta_theta}, inner={self.eta_in}")

        # Auto-epsilon: calibrate from first inner loop's forget loss
        # ε_fgt = multiplier × baseline_forget (forget must stay above this)
        inner_losses = self.inner_loop(device)
        baseline_fgt = sum(inner_losses) / len(inner_losses)
        self.epsilon_fgt = self.epsilon_multiplier * baseline_fgt
        logger.info(f"  Auto-ε_fgt: inner_fgt_avg={baseline_fgt:.4f}, "
                    f"multiplier={self.epsilon_multiplier}, ε_fgt={self.epsilon_fgt:.4f}")

        history = []
        log_every = max(1, max_outer_steps // 30)

        for t in range(max_outer_steps):
            t_start = time.time()

            if t > 0:
                inner_losses = self.inner_loop(device)

            L_fgt, L_ret, r = self.outer_step(device)
            dt = time.time() - t_start

            inner_mean = sum(inner_losses) / len(inner_losses) if inner_losses else 0.0
            history.append({
                "step": t, "L_fgt": L_fgt, "L_ret": L_ret,
                "r": r, "lambda": self.lambda_dual, "inner_fgt_mean": inner_mean, "dt": dt,
            })

            if t % log_every == 0 or t == max_outer_steps - 1:
                logger.info(
                    f"  [{t:4d}/{max_outer_steps}] "
                    f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                    f"r={r:+.4f} λ={self.lambda_dual:.3f} "
                    f"inner_fgt={inner_mean:.4f} dt={dt:.1f}s"
                )

            if L_ret > 10.0:
                logger.warning(f"  L_ret={L_ret:.1f} > 10.0 — collapsed at step {t+1}.")
                break

        # Merge and save
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

        self.evaluate()
