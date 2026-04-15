"""
GSP-SIBL: Gradient Subspace Partitioned Bilevel Unlearning
==========================================================

Bilevel LoRA unlearning with GSP-weighted sampling.

GSP pre-computes per-sample entanglement scores E_f (forget) and E_r (retain)
measuring gradient interference between forget and retain data.  Scores drive
weighted sampling:

  Forget sampling weight:  w_f[i] = 1 / (1 + α * E_f[i])   — down-sample entangled
  Retain sampling weight:  w_r[j] = 1 + β * E_r[j]          — up-sample entangled

Since MUSE uses PretrainingDataset (text chunking), GSP per-sample scores are
mapped to chunks proportionally, following the same pattern as SIBL.

Bilevel structure (forget_outer only — forget_inner is dead):
  Inner loop (K steps):  retain CE (GSP-weighted sampling)
  Outer loop (1 step):   forget logit_margin (GSP-weighted sampling) + ALM

No PerTA, no implicit correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
import os
import json
from typing import Optional

from torch.utils.data import DataLoader, WeightedRandomSampler

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class GspSIBL(UnlearnTrainer):

    def __init__(
        self,
        *args,
        # GSP
        gsp_interference_path: str = "saves/unlearn/gsp_interference_News.pt",
        gsp_alpha: float = 2.0,   # forget down-weighting strength
        gsp_beta: float = 2.0,    # retain up-weighting strength
        # LoRA
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # Bilevel structure
        T: int = -1,
        K: int = 3,
        eta_theta: float = 5e-5,
        eta_in: float = 2e-4,
        # ALM
        epsilon: float = 0.70,
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 5.0,
        # Forget loss
        forget_loss_type: str = "logit_margin",
        ga_clip: float = 1.0,
        # Recovery
        recovery_epochs: int = 0,
        recovery_lr: float = 1e-5,
        # Checkpointing
        checkpoint_every_epoch: bool = False,
        eval_at_steps: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # GSP
        self.gsp_interference_path = gsp_interference_path
        self.gsp_alpha = gsp_alpha
        self.gsp_beta = gsp_beta
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
        self.epsilon = epsilon
        self.rho = rho
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.lambda_dual = float(lambda_init)
        # Forget
        self.forget_loss_type = forget_loss_type
        self.ga_clip = ga_clip
        # Recovery
        self.recovery_epochs = recovery_epochs
        self.recovery_lr = recovery_lr
        # Checkpointing
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.eval_at_steps = set(eval_at_steps) if eval_at_steps else set()

    # ------------------------------------------------------------------
    # GSP weight loading + mapping to chunked dataset
    # ------------------------------------------------------------------
    def _load_gsp_weights(self):
        """Load GSP interference data and compute per-sample weights."""
        logger.info(f"Loading GSP data: {self.gsp_interference_path}")
        data = torch.load(self.gsp_interference_path, map_location="cpu", weights_only=True)
        E_f = data["E_f"].float()
        E_r = data["E_r"].float()

        # Forget weights: down-weight high-interference samples
        w_f = 1.0 / (1.0 + self.gsp_alpha * E_f)
        # Retain weights: up-weight high-interference samples
        w_r = 1.0 + self.gsp_beta * E_r

        logger.info(f"  E_f: mean={E_f.mean():.4f}, std={E_f.std():.4f}, "
                     f"min={E_f.min():.4f}, max={E_f.max():.4f}")
        logger.info(f"  E_r: mean={E_r.mean():.4f}, std={E_r.std():.4f}, "
                     f"min={E_r.min():.4f}, max={E_r.max():.4f}")
        logger.info(f"  w_f: [{w_f.min():.3f}, {w_f.max():.3f}], mean={w_f.mean():.3f}")
        logger.info(f"  w_r: [{w_r.min():.3f}, {w_r.max():.3f}], mean={w_r.mean():.3f}")

        self._gsp_w_f = w_f  # raw per-sample weights (n_forget,)
        self._gsp_w_r = w_r  # raw per-sample weights (n_retain,)

    def _adapt_weights_to_chunks(self, raw_weights, n_chunks):
        """Map per-sample GSP weights to PretrainingDataset chunks.

        PretrainingDataset concatenates all text and re-chunks into fixed-length
        pieces. Chunk i approximately corresponds to raw sample
        floor(i * n_raw / n_chunks). We map weights proportionally.
        """
        n_raw = len(raw_weights)
        if n_raw == n_chunks:
            return raw_weights.clone()

        adapted = torch.zeros(n_chunks)
        for i in range(n_chunks):
            src_idx = min(int(i * n_raw / n_chunks), n_raw - 1)
            adapted[i] = raw_weights[src_idx]

        logger.info(f"  Adapted {n_raw} raw weights -> {n_chunks} chunks, "
                     f"range=[{adapted.min():.3f}, {adapted.max():.3f}]")
        return adapted

    # ------------------------------------------------------------------
    # Custom dataloader with GSP-weighted sampling
    # ------------------------------------------------------------------
    def _make_gsp_dataloader(self):
        """Create a dataloader with GSP-weighted forget/retain sampling.

        We build separate forget and retain dataloaders with WeightedRandomSampler,
        then zip them in the training loop.
        """
        base_ds = self.train_dataset  # ForgetRetainDataset
        forget_ds = base_ds.forget
        retain_ds = base_ds.retain

        n_forget = len(forget_ds)
        n_retain = len(retain_ds)

        # Adapt GSP weights to chunk counts
        w_f = self._adapt_weights_to_chunks(self._gsp_w_f, n_forget)
        w_r = self._adapt_weights_to_chunks(self._gsp_w_r, n_retain)

        # Create weighted samplers
        forget_sampler = WeightedRandomSampler(
            weights=w_f.tolist(), num_samples=n_forget, replacement=True
        )
        retain_sampler = WeightedRandomSampler(
            weights=w_r.tolist(), num_samples=n_retain, replacement=True
        )

        bs = self.args.per_device_train_batch_size
        nw = self.args.dataloader_num_workers

        forget_loader = DataLoader(
            forget_ds, batch_size=bs, sampler=forget_sampler,
            collate_fn=self.data_collator, num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
        )
        retain_loader = DataLoader(
            retain_ds, batch_size=bs, sampler=retain_sampler,
            collate_fn=self.data_collator, num_workers=nw,
            pin_memory=self.args.dataloader_pin_memory,
        )

        logger.info(f"  GSP dataloaders: forget={n_forget} chunks, retain={n_retain} chunks, "
                     f"batch_size={bs}")

        return forget_loader, retain_loader

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------
    def _wrap_with_lora(self):
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha_val,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def _compute_ce_loss(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels", input_ids).to(device)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs.loss

    def _compute_logit_margin_loss(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        margins = logits.max(dim=-1)[0] - logits.mean(dim=-1)
        return margins.mean()

    def _compute_forget_loss(self, batch, device):
        if self.forget_loss_type == "logit_margin":
            return self._compute_logit_margin_loss(batch, device)
        elif self.forget_loss_type == "ga":
            return -self._compute_ce_loss(batch, device)
        else:
            raise ValueError(f"Unknown forget_loss_type: {self.forget_loss_type}")

    # ------------------------------------------------------------------
    # Inner / outer steps
    # ------------------------------------------------------------------
    def inner_step(self, retain_batch, device):
        """Inner loop: retain CE."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._inner_opt.zero_grad()
        loss = self._compute_ce_loss(retain_batch, device)
        loss.backward()
        self._inner_opt.step()
        return loss.item()

    def outer_step(self, forget_batch, retain_batch, device):
        """Outer step: forget + ALM on retain."""
        self.model.enable_adapter_layers()
        self.model.train()
        self._outer_opt.zero_grad()

        L_fgt = self._compute_forget_loss(forget_batch, device)
        L_ret = self._compute_ce_loss(retain_batch, device)

        r_val = max(0.0, L_ret.item() - self.epsilon)
        alm_coeff = self.lambda_dual + self.rho * r_val

        L_outer = L_fgt + alm_coeff * L_ret
        L_outer.backward()

        if self.forget_loss_type == "ga" and self.ga_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.ga_clip,
            )

        self._outer_opt.step()

        # Dual update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r_val)
        if self.lambda_max > 0:
            self.lambda_dual = min(self.lambda_dual, self.lambda_max)

        return L_fgt.item(), L_ret.item(), r_val

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        device = self.args.device

        logger.info("=" * 60)
        logger.info("GSP-SIBL: Gradient Subspace Partitioned Bilevel")
        logger.info("=" * 60)

        # Load GSP weights
        self._load_gsp_weights()

        # LoRA
        self._wrap_with_lora()

        # Optimizers (SGD for inner — avoids stale Adam momentum across bilevel)
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        self._inner_opt = torch.optim.SGD(lora_params, lr=self.eta_in)
        self._outer_opt = torch.optim.Adam(lora_params, lr=self.eta_theta)
        n_params = sum(p.numel() for p in lora_params)
        logger.info(f"Optimizers: inner=SGD lr={self.eta_in}, outer=Adam lr={self.eta_theta}, "
                     f"params={n_params:,}")

        # Gradient checkpointing
        if getattr(self.args, "gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # GSP-weighted dataloaders (separate forget/retain)
        forget_loader, retain_loader = self._make_gsp_dataloader()
        steps_per_epoch = min(len(forget_loader), len(retain_loader))
        num_epochs = max(1, int(self.args.num_train_epochs))
        max_steps = self.T if self.T > 0 else num_epochs * steps_per_epoch

        logger.info(f"  Epochs={num_epochs}, steps/epoch={steps_per_epoch}, "
                     f"max_steps={max_steps}, K={self.K}")
        logger.info(f"  Forget loss: {self.forget_loss_type}")
        logger.info(f"  GSP: α={self.gsp_alpha}, β={self.gsp_beta}")
        logger.info(f"  ALM: eps={self.epsilon}, rho={self.rho}, "
                     f"lambda={self.lambda_init}")
        if torch.cuda.is_available():
            logger.info(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        history = []
        global_step = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_fgt, epoch_ret = [], []

            for batch_idx, (forget_batch, retain_batch) in enumerate(
                zip(forget_loader, retain_loader)
            ):
                if global_step >= max_steps:
                    break

                t0 = time.time()

                # Inner loop: K steps of retain CE
                inner_losses = []
                for _ in range(self.K):
                    l_in = self.inner_step(retain_batch, device)
                    inner_losses.append(l_in)

                # Outer step: forget + ALM
                L_fgt, L_ret, r_val = self.outer_step(
                    forget_batch, retain_batch, device
                )

                dt = time.time() - t0

                step_info = {
                    "step": global_step, "epoch": epoch,
                    "L_fgt": L_fgt, "L_ret": L_ret, "r": r_val,
                    "lambda": self.lambda_dual,
                    "inner_mean": sum(inner_losses) / len(inner_losses),
                    "dt": dt,
                }
                history.append(step_info)
                epoch_fgt.append(L_fgt)
                epoch_ret.append(L_ret)

                log_every = max(1, steps_per_epoch // 10)
                if global_step % log_every == 0 or global_step == max_steps - 1:
                    logger.info(
                        f"  [{global_step:4d}/{max_steps}|e{epoch+1}] "
                        f"L_fgt={L_fgt:.4f} L_ret={L_ret:.4f} "
                        f"r={r_val:+.4f} lam={self.lambda_dual:.3f} "
                        f"inner={step_info['inner_mean']:.4f} "
                        f"olr={self.eta_theta:.2e} dt={dt:.1f}s"
                    )

                global_step += 1

                if global_step in self.eval_at_steps:
                    self._save_checkpoint(global_step, history)

                if L_ret > 10.0:
                    logger.warning(f"  L_ret={L_ret:.1f} > 10 — collapsed.")
                    break

            if epoch_fgt:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs}: {batch_idx+1} steps, "
                    f"mean_fgt={sum(epoch_fgt)/len(epoch_fgt):.4f}, "
                    f"mean_ret={sum(epoch_ret)/len(epoch_ret):.4f}, "
                    f"dt={time.time()-epoch_start:.0f}s"
                )

            if self.checkpoint_every_epoch and epoch < num_epochs - 1 and epoch_fgt:
                ckpt_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                logger.info(f"  Checkpoint: {ckpt_dir}")

            if global_step >= max_steps or L_ret > 10.0:
                break

        # Retain recovery
        if self.recovery_epochs > 0:
            logger.info("=" * 60)
            logger.info(f"Retain recovery: {self.recovery_epochs} epochs, "
                         f"lr={self.recovery_lr}")
            logger.info("=" * 60)

            recovery_opt = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.recovery_lr,
            )
            self.model.enable_adapter_layers()
            self.model.train()

            for rec_epoch in range(self.recovery_epochs):
                rec_start = time.time()
                rec_losses = []
                for retain_batch in retain_loader:
                    recovery_opt.zero_grad()
                    loss = self._compute_ce_loss(retain_batch, device)
                    loss.backward()
                    recovery_opt.step()
                    rec_losses.append(loss.item())
                mean_loss = sum(rec_losses) / len(rec_losses)
                logger.info(
                    f"  Recovery epoch {rec_epoch+1}/{self.recovery_epochs}: "
                    f"mean_ret={mean_loss:.4f}, dt={time.time()-rec_start:.0f}s"
                )

        # Merge and save
        logger.info("Merging LoRA into base model...")
        self.model = self.model.merge_and_unload()
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.model.save_pretrained(self.args.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.args.output_dir)

        with open(os.path.join(self.args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved to {self.args.output_dir}")
        if history:
            logger.info(f"Final: lam={self.lambda_dual:.3f}, "
                         f"L_fgt={history[-1]['L_fgt']:.4f}, "
                         f"L_ret={history[-1]['L_ret']:.4f}")

        self.evaluate()

    def _save_checkpoint(self, step, history):
        ckpt_dir = os.path.join(self.args.output_dir, f"step-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"  Saving checkpoint at step {step}...")
        self.model.eval()
        self.model.merge_adapter()
        peft_sd = self.model.base_model.model.state_dict()
        clean_sd = {k.replace(".base_layer", ""): v
                    for k, v in peft_sd.items() if "lora_" not in k}
        self.model.base_model.model.save_pretrained(ckpt_dir, state_dict=clean_sd)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir)
        self.model.unmerge_adapter()
        self.model.train()
        with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
