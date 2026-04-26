"""
LoKU: Low-Rank Knowledge Unlearning (Cha et al., ICLR 2025)

Ported from: https://github.com/csm9493/efficient-llm-unlearning
  - IHL loss: TOFU/dataloader.py lines 32-82 (copied verbatim)
  - FILA init: TOFU/forget.py lines 237-287 (copied verbatim)
  - Fisher importance: TOFU/measure_importance.py lines 214-263 (copied verbatim)
"""

import torch
from torch import Tensor, tensor
from typing import Optional
from functools import reduce
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from peft import LoraConfig, get_peft_model
from torchmetrics.utilities.data import to_onehot
from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_format
from torchmetrics.functional.classification.hinge import (
    _multiclass_hinge_loss_arg_validation,
    _multiclass_hinge_loss_tensor_validation,
    _hinge_loss_compute
)

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)


# ============================================================================
# IHL loss — copied from LoKU TOFU/dataloader.py lines 32-82
# ============================================================================
def _custom_multiclass_hinge_loss_update(
    preds,
    target,
    alpha,
    squared,
    multiclass_mode = "crammer-singer"
):
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)

    target = to_onehot(target, max(2, preds.shape[1])).bool()
    if multiclass_mode == "crammer-singer":
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
    else:
        target = target.bool()
        margin = torch.zeros_like(preds)
        margin[target] = preds[target]
        margin[~target] = -preds[~target]

    measures = alpha + margin
    measures = torch.clamp(measures, 0)

    if squared:
        measures = measures.pow(2)

    total = tensor(target.shape[0], device=target.device)
    return measures.sum(dim=0), total

def multiclass_hinge_loss(
    preds,
    target,
    num_classes,
    alpha = 1.0,
    squared = False,
    multiclass_mode = "crammer-singer",
    ignore_index = None,
    validate_args = True,
):
    if validate_args:
        _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        _multiclass_hinge_loss_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
    measures, total = _custom_multiclass_hinge_loss_update(
        preds,
        target,
        alpha,
        squared,
        multiclass_mode,
    )
    return _hinge_loss_compute(measures, total)


# ============================================================================
# Helper from LoKU TOFU/forget.py + measure_importance.py
# ============================================================================
def find_all_linear_names(model):
    """Copied from LoKU TOFU/forget.py lines 18-28"""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_module_by_name(module, access_string):
    """Copied from LoKU TOFU/forget.py line 238-240"""
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


# ============================================================================
# Fisher importance — copied from LoKU TOFU/measure_importance.py lines 206-263
# Adapted: uses our DataLoader format instead of their custom collator
# ============================================================================
def compute_fisher_importances(model, paired_dataloader, device):
    """Compute Fisher importances using paired forget/retain batches.

    Matches original measure_importance.py: iterates paired dataloader once,
    each batch yields (forget_batch, retain_batch).
    """
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Find all names of linear weights
    target_modules = find_all_linear_names(model)

    ##### GET IMPORTANCES
    importance_f = {}
    importance_r = {}
    for name, param in model.named_parameters():
        for t in target_modules:
            if t in name and 'weight' in name:
                importance_f[name] = 0
                importance_r[name] = 0

    f_cnt = 0
    r_cnt = 0
    for epochs in range(1): # use 1-epoch importance measurement for now
        for step, inputs in tqdm(enumerate(paired_dataloader), desc="Fisher importance"):

            # Forget
            forget_input = inputs["forget"]
            input_ids = forget_input["input_ids"].to(device)
            labels = forget_input["labels"].to(device)
            attention_mask = forget_input["attention_mask"].to(device)
            output = model(input_ids, labels=labels, attention_mask=attention_mask)
            output.loss.backward()
            cnt = torch.sum(labels != -100)
            for n, lp in model.named_parameters():
                if n in importance_f:
                    importance_f[n] += (lp.grad.pow(2) * cnt).detach().cpu()
                lp.grad = None
            f_cnt += cnt

            # Retain
            retain_input = inputs["retain"]
            input_ids = retain_input["input_ids"].to(device)
            labels = retain_input["labels"].to(device)
            attention_mask = retain_input["attention_mask"].to(device)
            output = model(input_ids, labels=labels, attention_mask=attention_mask)
            output.loss.backward()
            cnt = torch.sum(labels != -100)
            for n, lp in model.named_parameters():
                if n in importance_r:
                    importance_r[n] += (lp.grad.pow(2) * cnt).detach().cpu()
                lp.grad = None
            r_cnt += cnt

    return importance_f, importance_r, f_cnt, r_cnt


# ============================================================================
# FILA init — copied from LoKU TOFU/forget.py lines 242-287
# ============================================================================
def apply_fila(model, importance_f, importance_r, f_cnt, r_cnt, lora_targets, lora_r):
    f_cnt = f_cnt.cpu()
    r_cnt = r_cnt.cpu()
    importances = {n: torch.div(importance_f[n]/f_cnt, 1e-5+(importance_r[n]/r_cnt)) for n in importance_f.keys()}

    applied_count = 0
    for old_name, importance in importances.items():

        if not any([target_name in old_name for target_name in lora_targets]):
            continue
        name = old_name.replace("module.", '')
        lora_A = 'base_model.model.'+name.replace(".weight", '')+'.lora_A'
        lora_B = 'base_model.model.'+name.replace(".weight", '')+'.lora_B'
        base_layer = 'base_model.model.'+name.replace(".weight", '')+'.base_layer'
        scaling = 'base_model.model.'+name.replace(".weight", '')+'.scaling'

        try:
            lora_A = get_module_by_name(model, lora_A)
            lora_B = get_module_by_name(model, lora_B)
            base_layer = get_module_by_name(model, base_layer)
            scaling = get_module_by_name(model, scaling)
        except AttributeError:
            continue

        orig_shape = base_layer.weight.shape
        W = base_layer.weight.data.reshape(orig_shape)
        dtype = W.dtype
        W = W.to(torch.float32)

        # Solve row-wise weighted low-rank approximation
        row_importance = importance.sum(dim=1).sqrt().to(W.device) # row-wise sum
        U, S, V = torch.svd_lowrank(row_importance[:,None] * W, q=lora_r)

        S = S / scaling['default']

        new_lora_A = (V * torch.sqrt(S)).t()
        new_lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S))
        new_residual = base_layer.weight.data.reshape(orig_shape) - scaling['default'] * new_lora_B @ new_lora_A

        lora_A['default'].weight.data = new_lora_A.contiguous().to(dtype)
        lora_B['default'].weight.data = new_lora_B.contiguous().to(dtype)
        base_layer.weight.data = new_residual.contiguous().to(dtype)
        applied_count += 1

    logger.info(f"FILA applied to {applied_count} layers")


# ============================================================================
# LoKU Trainer
# ============================================================================
class LoKU(UnlearnTrainer):

    def __init__(
        self,
        *args,
        # IHL
        margin_alpha: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 1.0,
        # LoRA (LoKU run_forget.sh: r=32, alpha=64, dropout=0, targets=all)
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.0,
        lora_target_modules: Optional[list] = None,
        # FILA
        use_fila: bool = True,
        fila_batch_size: int = 4,
        # retain loss
        retain_loss_type: str = "NLL",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.margin_alpha = margin_alpha
        self.gamma = gamma
        self.alpha_coeff = alpha
        self.lora_r = lora_r
        self.lora_alpha_val = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]
        self.use_fila = use_fila
        self.fila_batch_size = fila_batch_size
        self.retain_loss_type = retain_loss_type

    # ---- IHL compute_loss — adapted from LoKU TOFU/dataloader.py lines 185-206 ----
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forget: IHL loss
        forget_inputs = inputs["forget"]
        input_ids = forget_inputs["input_ids"]
        labels = forget_inputs["labels"]
        attention_mask = forget_inputs["attention_mask"]

        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

        scores = outputs.logits
        shift_logits = scores[..., :-1, :].contiguous().squeeze().view(-1, scores.size(-1)) # [BN, V]
        shift_labels = labels[..., 1:].contiguous().squeeze().view(-1) # [BN,]
        forget_loss = multiclass_hinge_loss(
            shift_logits[shift_labels != -100,:], # ignore pad tokens
            shift_labels[shift_labels != -100],
            shift_logits.size(-1),
            alpha=self.margin_alpha,
        )

        # Retain: CE loss
        retain_inputs = inputs["retain"]
        retain_outputs = model(
            retain_inputs["input_ids"],
            labels=retain_inputs["labels"],
            attention_mask=retain_inputs["attention_mask"],
        )
        retain_loss = retain_outputs.loss

        loss = self.gamma * forget_loss + self.alpha_coeff * retain_loss
        return (loss, outputs) if return_outputs else loss

    def train(self, **kwargs):
        device = self.args.device

        # Step 1: Compute Fisher on base model BEFORE LoRA
        # Uses paired dataloader (same as original measure_importance.py)
        importance_f, importance_r, f_cnt, r_cnt = None, None, None, None
        if self.use_fila:
            logger.info("Computing Fisher importances on base model...")
            paired_dl = DataLoader(
                self.train_dataset, batch_size=self.fila_batch_size,
                shuffle=False, collate_fn=self.data_collator, drop_last=False,
            )
            importance_f, importance_r, f_cnt, r_cnt = compute_fisher_importances(
                self.model, paired_dl, device
            )
            self.model.zero_grad()
            logger.info("Fisher importances computed.")

        # Step 2: Wrap model with LoRA (from LoKU forget.py lines 225-235)
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha_val,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

        # Step 3: Apply FILA (from LoKU forget.py lines 242-287)
        if self.use_fila and importance_f is not None:
            logger.info("Applying FILA initialization...")
            apply_fila(
                self.model, importance_f, importance_r, f_cnt, r_cnt,
                self.lora_target_modules, self.lora_r,
            )
            del importance_f, importance_r
            torch.cuda.empty_cache()
            logger.info("FILA initialization complete.")

        # Step 4: Train with IHL + retain CE
        self.model.config.use_cache = False
        return super().train(**kwargs)
