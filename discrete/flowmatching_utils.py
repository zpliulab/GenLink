import torch
from torch.nn import functional as F
import numpy as np
from discrete.network_preprocess import PlaceHolder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score

def Evaluation(y_true, y_pred, flag=False):
    if isinstance(y_pred, torch.Tensor):
        y_p = y_pred.detach().cpu().numpy()
    else:
        y_p = np.array(y_pred)
    if isinstance(y_true, torch.Tensor):
        y_t = y_true.detach().cpu().numpy()
    else:
        y_t = np.array(y_true)
    y_p = y_p.flatten()
    y_t = y_t.flatten()
    auroc = roc_auc_score(y_t, y_p)
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_t, y_p)
    auprc = auc(recall_curve, precision_curve)
    f1_curve = 2 * precision_curve[:-1] * recall_curve[:-1] / (precision_curve[:-1] + recall_curve[:-1] + 1e-08)
    best_threshold = thresholds_pr[np.argmax(f1_curve)]
    y_pred_label = (y_p >= best_threshold).astype(int)
    f1 = f1_score(y_t, y_pred_label)
    precision_val = precision_score(y_t, y_pred_label, zero_division=0)
    return {'AUROC': auroc, 'AUPRC': auprc, 'F1-score': f1, 'precision': precision_val}

def sample_discrete_feature_noise(X, ATAC, limit_dist, node_mask, region_mask=None, seed=None):
    """ 对全噪声采样， 可用于测试阶段初始噪声，符合t=0时，分布全为噪声。"""
    bs, n_max = node_mask.shape
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    if seed is not None:
        torch.manual_seed(seed)
        U_E = e_limit.flatten(end_dim=-2).multinomial(num_samples=1).reshape(bs, n_max, n_max)
    else:
        U_E = e_limit.flatten(end_dim=-2).multinomial(num_samples=1).reshape(bs, n_max, n_max)
    U_E = delete_dig_from_batch(U_E)
    long_mask = node_mask.long()
    U_E = U_E.type_as(long_mask)
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
    y = torch.empty(1, 0).type_as(U_E)
    return PlaceHolder(X=X, ATAC=ATAC, E=U_E, y=y).mask(node_mask, region_mask)

def delete_dig_from_batch(batchdata):
    for i in range(0, batchdata.shape[0]):
        a = batchdata[i, :, :]
        a = a.fill_diagonal_(0)
        batchdata[i, :, :] = a
    return batchdata
