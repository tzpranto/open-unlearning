"""Forget loss functions for LoRA-BiAL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ce_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", input_ids).to(device)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


def compute_npo_loss(model, batch, device, npo_beta=4.0):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", input_ids).to(device)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    shifted_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    model.disable_adapter_layers()
    with torch.no_grad():
        ref_logits = model(**inputs).logits[..., :-1, :].contiguous()
        ref_nll = loss_fn(ref_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    model.enable_adapter_layers()
    model_logits = model(**inputs).logits[..., :-1, :].contiguous()
    model_nll = loss_fn(model_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    log_ratio = -(model_nll - ref_nll)
    return -2 / npo_beta * F.logsigmoid(npo_beta * (-log_ratio)).mean()


def compute_logit_margin_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    margins = logits.max(dim=-1)[0] - logits.mean(dim=-1)
    mask = attention_mask.float()
    return (margins * mask).sum() / mask.sum().clamp(min=1e-8)


def compute_clamped_entropy_loss(model, batch, device, tau=0.7):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    H = -(probs * log_probs).sum(dim=-1)

    H_max = torch.log(torch.tensor(float(logits.shape[-1]), device=device))
    per_token_loss = torch.clamp(tau * H_max - H, min=0.0)
    mask = attention_mask.float()
    return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)


FORGET_LOSS_DISPATCH = {
    "npo": lambda model, batch, device, **kw: compute_npo_loss(model, batch, device, kw.get("npo_beta", 4.0)),
    "ga": lambda model, batch, device, **kw: -compute_ce_loss(model, batch, device),
    "logit_margin": lambda model, batch, device, **kw: compute_logit_margin_loss(model, batch, device),
    "clamped_entropy": lambda model, batch, device, **kw: compute_clamped_entropy_loss(model, batch, device, kw.get("clamped_entropy_tau", 0.7)),
}
