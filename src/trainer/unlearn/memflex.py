import copy
import json
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


class MemFlex(UnlearnTrainer):
    """MemFlex: Knowledge Localization + Weighted GA+Descent.

    From "To Forget or Not? Towards Practical Knowledge Unlearning for LLMs" (EMNLP 2024).

    Phase 1 (localization): Identifies LoRA parameters responsible for forget knowledge
    by comparing gradient directions on forget vs retain data with random labels.

    Phase 2 (unlearning): Weighted GA+Descent on located parameters only.
    - Forget loss weight: -forget_weight (gradient ascent)
    - Retain loss weight: +retain_weight (gradient descent)
    """

    def __init__(
        self,
        forget_weight=0.4,
        retain_weight=2.0,
        mu=0.92,
        sigma=6e-4,
        num_random_labels=5,
        located_params_path=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        self.mu = mu
        self.sigma = sigma
        self.num_random_labels = num_random_labels

        if located_params_path:
            self._load_located_params(located_params_path)
        else:
            logger.info(
                "No located_params_path provided. All parameters will be trainable. "
                "Run localization first with MemFlexLocalize, then pass the path."
            )

    def _load_located_params(self, path):
        with open(path, "r") as f:
            located_names = json.load(f)
        logger.info(f"MemFlex: Loading {len(located_names)} located parameter names from {path}")

        frozen_count = 0
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if name in located_names:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"MemFlex: {unfrozen_count} params unfrozen, {frozen_count} params frozen")

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        forget_loss = -self.forget_weight * forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_outputs = model(**retain_inputs)
        retain_loss = self.retain_weight * retain_outputs.loss

        loss = forget_loss + retain_loss

        return (loss, forget_outputs) if return_outputs else loss


class MemFlexLocalize:
    """Pre-computation step for MemFlex: gradient-based parameter localization.

    Computes gradient information matrices on forget and retain data with random labels,
    then selects parameters where gradient directions diverge (cos < mu) and forget
    gradient magnitude is significant (|g| > sigma).
    """

    def __init__(self, model, tokenizer, forget_dataset, retain_dataset, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.device = device

    def _compute_gradient_info(self, dataset, num_copies=5):
        """Compute average gradient magnitude per parameter using random labels."""
        self.model.eval()
        grad_accum = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_accum[name] = torch.zeros_like(param.data)

        collator = torch.utils.data.dataloader.default_collate
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        for copy_idx in range(num_copies):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Replace non-ignored label tokens with random tokens
                valid_mask = labels != -100
                random_labels = torch.randint(
                    0, self.tokenizer.vocab_size, labels.shape, device=self.device
                )
                labels = torch.where(valid_mask, random_labels, labels)

                self.model.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                outputs.loss.backward()

                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_accum[name] += param.grad.detach().abs()

        num_total = num_copies * len(dataset)
        for name in grad_accum:
            grad_accum[name] /= num_total

        return grad_accum

    def localize(self, mu=0.92, sigma=6e-4, output_path=None):
        """Run localization and return list of parameter names to unfreeze."""
        logger.info("MemFlex Localization: Computing forget gradients...")
        forget_grads = self._compute_gradient_info(self.forget_dataset)

        logger.info("MemFlex Localization: Computing retain gradients...")
        retain_grads = self._compute_gradient_info(self.retain_dataset)

        located_params = []
        for name in forget_grads:
            fg = forget_grads[name].flatten().float()
            rg = retain_grads[name].flatten().float()

            cos_sim = F.cosine_similarity(fg.unsqueeze(0), rg.unsqueeze(0)).item()
            grad_mag = fg.mean().item()

            if cos_sim < mu and grad_mag > sigma:
                located_params.append(name)

        logger.info(
            f"MemFlex Localization: {len(located_params)}/{len(forget_grads)} "
            f"parameters selected (mu={mu}, sigma={sigma})"
        )

        if output_path:
            with open(output_path, "w") as f:
                json.dump(located_params, f, indent=2)
            logger.info(f"MemFlex Localization: Saved to {output_path}")

        return located_params
