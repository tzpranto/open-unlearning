"""
Loss Functions for S-BiAL Unlearning
=====================================

Forget losses: logit_margin, grad_ascent, npo
Regularization: l1, l2, none
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forget losses
# ---------------------------------------------------------------------------

def logit_margin(model, batch, device, **kwargs):
    """Logit margin flattening: minimize max(logits) - mean(logits)."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    return (logits.max(dim=-1)[0] - logits.mean(dim=-1)).mean()


def grad_ascent(model, batch, device, **kwargs):
    """Gradient ascent: negative cross-entropy."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", input_ids).to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return -outputs.loss


def npo(model, batch, device, ref_model=None, beta=1.0, **kwargs):
    """Negative Preference Optimization (DPO-style forget loss)."""
    if ref_model is None:
        logger.warning("NPO requires ref_model; falling back to grad_ascent")
        return grad_ascent(model, batch, device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", input_ids).to(device)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    shifted_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    # Model NLL
    logits = model(**inputs).logits[..., :-1, :].contiguous()
    model_nll = loss_fn(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    # Reference NLL
    with torch.no_grad():
        ref_logits = ref_model(**inputs).logits[..., :-1, :].contiguous()
        ref_nll = loss_fn(ref_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    log_ratio = -(model_nll - ref_nll)
    return -2.0 / beta * F.logsigmoid(beta * (-log_ratio)).mean()


# ---------------------------------------------------------------------------
# Regularization
# ---------------------------------------------------------------------------

def reg_l1(model, mask_dict, gamma, device):
    """L1 regularization (sparsity-inducing)."""
    reg = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in mask_dict:
            reg = reg + (param.abs() * mask_dict[name]).sum()
    return gamma * reg


def reg_l2(model, mask_dict, gamma, device):
    """L2 regularization (weight decay)."""
    reg = torch.tensor(0.0, device=device)
    for name, param in model.named_parameters():
        if name in mask_dict:
            reg = reg + ((param ** 2) * mask_dict[name]).sum()
    return gamma * reg


def reg_none(model, mask_dict, gamma, device):
    """No regularization."""
    return torch.tensor(0.0, device=device)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

AVAILABLE_FORGET_LOSSES = ["logit_margin", "grad_ascent", "npo"]
AVAILABLE_REGULARIZATIONS = ["l1", "l2", "none"]

_FORGET_LOSS_MAP = {
    "logit_margin": logit_margin,
    "grad_ascent": grad_ascent,
    "npo": npo,
}

_REGULARIZATION_MAP = {
    "l1": reg_l1,
    "l2": reg_l2,
    "none": reg_none,
}


def get_forget_loss_fn(loss_type: str):
    if loss_type not in _FORGET_LOSS_MAP:
        raise ValueError(
            f"Unknown forget loss: {loss_type}. Available: {AVAILABLE_FORGET_LOSSES}"
        )
    return _FORGET_LOSS_MAP[loss_type]


def get_regularization_fn(reg_type: str):
    if reg_type not in _REGULARIZATION_MAP:
        raise ValueError(
            f"Unknown regularization: {reg_type}. Available: {AVAILABLE_REGULARIZATIONS}"
        )
    return _REGULARIZATION_MAP[reg_type]
