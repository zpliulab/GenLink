import torch
import torch.nn as nn
from torch.nn import functional as F

class TrainLossDiscrete(nn.Module):

    def __init__(self, lambda_train=0.1):
        super().__init__()

    def forward(self, masked_pred_E, nosoft_pred_E, true_E, CEloss=True):
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))
        mask_E = (true_E != 0.0).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        num_pos = flat_true_E[:, 1].sum()
        num_neg = flat_true_E[:, 0].sum()
        pos_weight_val = (num_neg / (num_pos + 1e-06)).clamp(max=20.0)
        flat_pred_logits = torch.reshape(nosoft_pred_E, (-1, nosoft_pred_E.size(-1)))[mask_E, :]
        target_indices = torch.argmax(flat_true_E, dim=1)
        class_weights = torch.tensor([1.0, pos_weight_val.item()], device=true_E.device)
        loss = F.cross_entropy(flat_pred_logits, target_indices, weight=class_weights)
        return loss
