"""
LoRA-BiAL: LoRA-based Bilevel Augmented Lagrangian for LLM Unlearning
=====================================================================

Bilevel optimization with ALM constraint:
  Inner loop (K SGD steps): minimize retain CE on LoRA params
  Outer loop (1 Adam step): minimize forget loss subject to retain constraint
  ALM: L_fgt + λ·(L_ret - ε) + ρ/2·max(0, L_ret - ε)²

Zero-overhead ref model: disable LoRA adapters → base model serves as reference.
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


class LoRABiAL(UnlearnTrainer):

    def __init__(
        self,
        *args,
        # LoRA parameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # Bilevel parameters (stage 3)
        T: int = -1,
        K: int = 3,
        eta_theta: float = 3e-5,
        eta_in: float = 2e-4,
        # ALM parameters
        epsilon_multiplier: float = 1.15,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 0.0,
        lambda_min: float = 0.1,
        dual_decay_factor: float = 0.1,
        # Forget loss
        forget_loss_type: str = "clamped_entropy",
        npo_beta: float = 4.0,
        clamped_entropy_tau: float = 0.7,
        # Training
        lr_schedule: str = "constant",
        warmup_fraction: float = 0.0,
        checkpoint_every_epoch: bool = False,
        checkpoint_every_n_steps: int = 0,
        eval_at_steps: Optional[list] = None,
        # LoRA init (for sequential unlearning)
        lora_init_path: Optional[str] = None,
        save_lora_only: bool = False,
        # Batching
        gradient_accumulation_steps: int = 8,
        inner_accumulation_steps: int = 0,
        inner_warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # LoRA
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        # Bilevel
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        # ALM
        self.epsilon = None  # set by auto-calibration at step 0
        self.epsilon_multiplier = epsilon_multiplier
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.dual_decay_factor = dual_decay_factor
        self.lambda_dual = float(lambda_init)
        # Forget loss
        self.forget_loss_type = forget_loss_type
        if forget_loss_type not in FORGET_LOSS_DISPATCH:
            raise ValueError(f"Unknown forget_loss_type: {forget_loss_type}. "
                             f"Available: {list(FORGET_LOSS_DISPATCH.keys())}")
        self.npo_beta = npo_beta
        self.clamped_entropy_tau = clamped_entropy_tau
        # Training
        self.lr_schedule = lr_schedule
        self.warmup_fraction = warmup_fraction
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else set()
        # LoRA init
        self.lora_init_path = lora_init_path
        self.save_lora_only = save_lora_only
        # Batching
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.inner_accumulation_steps = inner_accumulation_steps if inner_accumulation_steps > 0 else gradient_accumulation_steps
        self.inner_warmup_steps = inner_warmup_steps
        self.max_grad_norm = max_grad_norm

    # ------------------------------------------------------------------
    # LoRA wrapping
    # ------------------------------------------------------------------
    def _wrap_with_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel
        if self.lora_init_path and os.path.isdir(self.lora_init_path):
            logger.info(f"Loading LoRA adapters from: {self.lora_init_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_init_path, is_trainable=True)
        else:
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

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------
    def _compute_ce_loss(self, batch, device):
        return compute_ce_loss(self.model, batch, device)

    def _compute_forget_loss(self, batch, device):
        return FORGET_LOSS_DISPATCH[self.forget_loss_type](
            self.model, batch, device,
            npo_beta=self.npo_beta,
            clamped_entropy_tau=self.clamped_entropy_tau,
        )

    # ------------------------------------------------------------------
    # Batch iterators
    # ------------------------------------------------------------------
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
    # Inner loop
    # ------------------------------------------------------------------
    def inner_loop(self, device):
        self.model.enable_adapter_layers()
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

    # ------------------------------------------------------------------
    # Outer step
    # ------------------------------------------------------------------
    def outer_step(self, device, global_step=0):
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

    # ------------------------------------------------------------------
    # Checkpoint saving (shared logic)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, ckpt_dir, history):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.eval()
        self.model.merge_adapter()
        peft_sd = self.model.base_model.model.state_dict()
        clean_sd = {
            k.replace(".base_layer", ""): v
            for k, v in peft_sd.items() if "lora_" not in k.replace(".base_layer", "")
        }
        self.model.base_model.model.save_pretrained(ckpt_dir, state_dict=clean_sd)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)
        self.model.unmerge_adapter()
        self.model.train()
        with open(os.path.join(ckpt_dir, "lora_bial_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        device = self.args.device

        # LoRA adapters
        logger.info("=" * 60)
        logger.info("LoRA adapters")
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
        logger.info(f"  ALM: ε_mul={self.epsilon_multiplier}, ρ={self.rho}, λ_init={self.lambda_init}"
                     f"{f', λ_max={self.lambda_max}' if self.lambda_max > 0 else ''}")
        logger.info(f"  LR: outer={self.eta_theta}, inner={self.eta_in}, schedule={self.lr_schedule}")
        logger.info(f"  LoRA: r={self.lora_r}, alpha={self.lora_alpha_val}")
        if self.inner_warmup_steps > 0:
            logger.info(f"  Inner warmup: outer-only for first {self.inner_warmup_steps} steps")
        if torch.cuda.is_available():
            logger.info(f"  GPU memory before bilevel: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # Training loop
        history = []
        log_every = max(1, max_outer_steps // 20)

        # Auto-epsilon: set ε from first inner loop's average retain loss
        inner_losses = self.inner_loop(device)
        baseline_ret = sum(inner_losses) / len(inner_losses)
        self.epsilon = self.epsilon_multiplier * baseline_ret
        logger.info(f"  Auto-ε: inner_avg={baseline_ret:.4f}, "
                    f"multiplier={self.epsilon_multiplier}, ε={self.epsilon:.4f}")

        for t in range(max_outer_steps):
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

            # Adaptive inner recovery when retain spikes
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
            history.append({
                "step": t, "epoch": epoch, "L_fgt": L_fgt, "L_ret": L_ret,
                "r": r, "lambda": self.lambda_dual, "inner_loss_mean": inner_mean,
                "dt": dt,
            })

            # Logging
            if t % log_every == 0 or t == max_outer_steps - 1:
                extras = ""
                if self._outer_scheduler is not None:
                    extras += f" olr={self._outer_opt.param_groups[0]['lr']:.2e}"
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

            # Periodic step checkpoint (LoRA only, for crash recovery)
            if (self.checkpoint_every_n_steps > 0
                    and global_step % self.checkpoint_every_n_steps == 0
                    and global_step < max_outer_steps):
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-step{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                logger.info(f"  LoRA checkpoint saved: {ckpt_dir}")

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

        # Save LoRA adapters separately (for sequential chaining)
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.save_lora_only:
            lora_dir = os.path.join(output_dir, "lora_adapters")
            os.makedirs(lora_dir, exist_ok=True)
            self.model.save_pretrained(lora_dir)
            logger.info(f"LoRA adapters saved to {lora_dir}")

        # Merge LoRA and save final model
        logger.info("Merging LoRA adapters into base model...")
        self.model = self.model.merge_and_unload()
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
