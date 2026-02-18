"""
Loss Functions Module for SIBL
==============================

This module provides modular loss functions and regularization methods
that can be used with SIBL (Sparse Bilevel Augmented Lagrangian) unlearning.

Supported Forget Loss Functions:
- logit_margin: Original SIBL loss (max_logits - mean_logits)
- grad_ascent: Negative cross-entropy (gradient ascent)
- grad_diff: Negative cross-entropy (same as grad_ascent for forget)
- npo: Negative Preference Optimization using DPO loss
- simnpo: Simplified NPO without reference model
- pdu: PDU-style squared margin loss with masking
- rmu: RMU-style activation steering loss (not fully supported, requires special setup)

Supported Regularization Methods:
- l1: L1 norm (sparsity-inducing)
- l2: L2 norm (weight decay)
- elastic_net: Combination of L1 and L2
- none: No regularization
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ForgetLossFunctions:
    """Collection of forget loss functions for unlearning."""

    @staticmethod
    def logit_margin(model, batch, device, **kwargs) -> torch.Tensor:
        """
        Original SIBL logit margin flattening loss.
        Aims to flatten the logit distribution by minimizing max - mean logits.

        Args:
            model: The model to compute loss for
            batch: Input batch containing input_ids and attention_mask
            device: Device to use

        Returns:
            Loss tensor
        """
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)
        margins = max_logits - mean_logits
        return margins.mean()

    @staticmethod
    def grad_ascent(model, batch, device, **kwargs) -> torch.Tensor:
        """
        Gradient ascent loss: negative cross-entropy.
        Maximizes the loss on forget set to unlearn.

        Args:
            model: The model to compute loss for
            batch: Input batch containing input_ids, attention_mask, labels
            device: Device to use

        Returns:
            Negative loss tensor (for gradient ascent)
        """
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', input_ids).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return -outputs.loss

    @staticmethod
    def grad_diff(model, batch, device, **kwargs) -> torch.Tensor:
        """
        Gradient difference loss for forget set (same as grad_ascent for forget part).

        Args:
            model: The model to compute loss for
            batch: Input batch
            device: Device to use

        Returns:
            Negative loss tensor
        """
        return ForgetLossFunctions.grad_ascent(model, batch, device, **kwargs)

    @staticmethod
    def npo(model, batch, device, ref_model=None, beta=1.0, **kwargs) -> torch.Tensor:
        """
        Negative Preference Optimization loss using DPO-style computation.

        Args:
            model: The model to compute loss for
            batch: Input batch
            device: Device to use
            ref_model: Reference model (required for NPO)
            beta: Temperature parameter for DPO loss

        Returns:
            NPO loss tensor
        """
        if ref_model is None:
            logger.warning("NPO requires ref_model but none provided. Falling back to grad_ascent.")
            return ForgetLossFunctions.grad_ascent(model, batch, device, **kwargs)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', input_ids).to(device)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        # Compute batch NLL for model
        outputs = model(**inputs)
        logits = outputs.logits
        shifted_labels = labels[..., 1:].contiguous()
        shifted_logits = logits[..., :-1, :].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        model_nll = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

        # Compute batch NLL for ref_model
        with torch.no_grad():
            ref_outputs = ref_model(**inputs)
            ref_logits = ref_outputs.logits
            ref_shifted_logits = ref_logits[..., :-1, :].contiguous()
            ref_nll = loss_fn(ref_shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

        # NPO: negative log ratio
        log_ratio = -(model_nll - ref_nll)
        loss = -2 / beta * F.logsigmoid(beta * (-log_ratio)).mean()

        return loss

    @staticmethod
    def simnpo(model, batch, device, delta=0.0, beta=1.0, **kwargs) -> torch.Tensor:
        """
        Simplified NPO loss without reference model.

        Args:
            model: The model to compute loss for
            batch: Input batch
            device: Device to use
            delta: Offset parameter
            beta: Temperature parameter

        Returns:
            SimNPO loss tensor
        """
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', input_ids).to(device)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        # Compute batch NLL
        outputs = model(**inputs)
        logits = outputs.logits
        shifted_labels = labels[..., 1:].contiguous()
        shifted_logits = logits[..., :-1, :].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        nll = loss_fn(shifted_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

        # SimNPO computation
        loss_mask = labels != -100
        forget_loss = nll / loss_mask.sum(-1) - delta
        forget_loss = -F.logsigmoid(beta * forget_loss).mean() * 2 / beta

        return forget_loss

    @staticmethod
    def pdu(model, batch, device, **kwargs) -> torch.Tensor:
        """
        PDU-style squared margin loss with masking.

        Args:
            model: The model to compute loss for
            batch: Input batch
            device: Device to use

        Returns:
            PDU loss tensor
        """
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', input_ids).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        logits = logits.reshape(-1, logits.size(-1))
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)

        # Squared margin loss
        forget_loss = (max_logits - mean_logits) ** 2
        mask = (labels != -100).reshape(-1)
        forget_loss = (forget_loss * mask).sum() / (mask.sum() + 1e-8)

        return forget_loss

    @staticmethod
    def rmu(model, batch, device, steering_coeff=20.0, **kwargs) -> torch.Tensor:
        """
        RMU-style loss - simplified version using logit margin.
        Full RMU requires module hooks and activation steering which is complex.
        This falls back to a PDU-like loss for simplicity.

        Args:
            model: The model to compute loss for
            batch: Input batch
            device: Device to use
            steering_coeff: Steering coefficient (not used in simplified version)

        Returns:
            RMU-like loss tensor
        """
        # RMU requires special activation hooks and steering vectors
        # For SIBL integration, we use a simplified version similar to PDU
        logger.debug("RMU in SIBL uses simplified logit-based loss")
        return ForgetLossFunctions.pdu(model, batch, device, **kwargs)


class RegularizationFunctions:
    """Collection of regularization methods."""

    @staticmethod
    def l1(model, mask_dict: Dict[str, torch.Tensor], gamma: float, device) -> torch.Tensor:
        """
        L1 regularization (promotes sparsity).

        Args:
            model: The model
            mask_dict: Dictionary of parameter masks
            gamma: Regularization coefficient
            device: Device to use

        Returns:
            L1 regularization term
        """
        reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in mask_dict:
                mask = mask_dict[name]
                reg = reg + (param.abs() * mask).sum()
        return gamma * reg

    @staticmethod
    def l2(model, mask_dict: Dict[str, torch.Tensor], gamma: float, device) -> torch.Tensor:
        """
        L2 regularization (weight decay).

        Args:
            model: The model
            mask_dict: Dictionary of parameter masks
            gamma: Regularization coefficient
            device: Device to use

        Returns:
            L2 regularization term
        """
        reg = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in mask_dict:
                mask = mask_dict[name]
                reg = reg + ((param ** 2) * mask).sum()
        return gamma * reg

    @staticmethod
    def elastic_net(model, mask_dict: Dict[str, torch.Tensor], gamma: float, device,
                    l1_ratio: float = 0.5) -> torch.Tensor:
        """
        Elastic net regularization (combination of L1 and L2).

        Args:
            model: The model
            mask_dict: Dictionary of parameter masks
            gamma: Regularization coefficient
            device: Device to use
            l1_ratio: Ratio of L1 vs L2 (0.5 = equal weight)

        Returns:
            Elastic net regularization term
        """
        l1_term = RegularizationFunctions.l1(model, mask_dict, 1.0, device)
        l2_term = RegularizationFunctions.l2(model, mask_dict, 1.0, device)
        return gamma * (l1_ratio * l1_term + (1 - l1_ratio) * l2_term)

    @staticmethod
    def none(model, mask_dict: Dict[str, torch.Tensor], gamma: float, device) -> torch.Tensor:
        """
        No regularization.

        Returns:
            Zero tensor
        """
        return torch.tensor(0.0, device=device)


def get_forget_loss_fn(loss_type: str):
    """
    Get the forget loss function by name.

    Args:
        loss_type: Name of the loss function

    Returns:
        Loss function callable

    Raises:
        ValueError: If loss type is not supported
    """
    loss_functions = {
        'logit_margin': ForgetLossFunctions.logit_margin,
        'grad_ascent': ForgetLossFunctions.grad_ascent,
        'grad_diff': ForgetLossFunctions.grad_diff,
        'npo': ForgetLossFunctions.npo,
        'simnpo': ForgetLossFunctions.simnpo,
        'pdu': ForgetLossFunctions.pdu,
        'rmu': ForgetLossFunctions.rmu,
    }

    if loss_type not in loss_functions:
        raise ValueError(
            f"Unknown forget loss type: {loss_type}. "
            f"Supported types: {list(loss_functions.keys())}"
        )

    return loss_functions[loss_type]


def get_regularization_fn(reg_type: str):
    """
    Get the regularization function by name.

    Args:
        reg_type: Name of the regularization method

    Returns:
        Regularization function callable

    Raises:
        ValueError: If regularization type is not supported
    """
    reg_functions = {
        'l1': RegularizationFunctions.l1,
        'l2': RegularizationFunctions.l2,
        'elastic_net': RegularizationFunctions.elastic_net,
        'none': RegularizationFunctions.none,
    }

    if reg_type not in reg_functions:
        raise ValueError(
            f"Unknown regularization type: {reg_type}. "
            f"Supported types: {list(reg_functions.keys())}"
        )

    return reg_functions[reg_type]


# List of available loss types for external reference
AVAILABLE_FORGET_LOSSES = ['logit_margin', 'grad_ascent', 'grad_diff', 'npo', 'simnpo', 'pdu', 'rmu']
AVAILABLE_REGULARIZATIONS = ['l1', 'l2', 'elastic_net', 'none']