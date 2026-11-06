from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from discrete.network_preprocess import PlaceHolder
from discrete import network_preprocess
from discrete import flowmatching_utils
from discrete.train_metrics import TrainLossDiscrete
from discrete.flowmatching_utils import Evaluation
import copy
import pandas as pd
from flowmatchingpackage.flow_matching import flow_matching_utils
from flowmatchingpackage.flow_matching.utils import p_xt_g_x1
from flowmatchingpackage.flow_matching.rate_matrix import RateMatrixDesigner
from make_final_net import cal_final_net2

class FlowMatching1D(nn.Module):

    def __init__(self, model, *, edge_percent=0.35, max_num_nodes=None, device='cpu', forward_steps=30, backward_steps=1000):
        super().__init__()
        self.device = device
        self.train_loss = TrainLossDiscrete()
        self.model = model
        self.forward_steps = forward_steps
        self.backward_step = backward_steps
        e_marginals = torch.tensor([1 - edge_percent, edge_percent], dtype=torch.float32, device=device)
        self.limit_dist = PlaceHolder(X=None, E=e_marginals, y=None)
        self.rate_matrix_designer = RateMatrixDesigner(limit_dist=self.limit_dist)

    def pred_shape2data(self, pred, noisy_data):
        bs, n, _, c = noisy_data['E_t'].shape
        pred.E = pred.E.reshape(bs, n, n, c)
        return network_preprocess.PlaceHolder(X=pred.X, ATAC=pred.ATAC, E=pred.E, y=pred.y)

    def p_losses(self, E, noisy_data, node_mask, calAUC=True):
        torch.set_grad_enabled(True)
        pred = self.model(node_mask, noisy_data)
        pred = self.pred_shape2data(pred, noisy_data)
        mask_pred_probs_E = pred.mask(node_mask, noisy_data['region_mask'])
        target = E
        l1 = self.train_loss(masked_pred_E=None, nosoft_pred_E=mask_pred_probs_E.E, true_E=target)
        t = noisy_data['t']
        loss = l1
        if calAUC:
            pred_E_soft = F.softmax(mask_pred_probs_E.E, dim=-1)
            AUC_list = []
            for bsi in range(E.shape[0]):
                true1 = E[bsi].argmax(dim=-1)
                true1 = true1[:, node_mask[bsi, :]]
                true1 = true1[node_mask[bsi, :], :]
                true1 = true1.flatten()
                pred1 = pred_E_soft[bsi, :, :, 1]
                pred1 = pred1[:, node_mask[bsi, :]]
                pred1 = pred1[node_mask[bsi, :], :]
                pred1 = pred1.flatten()
                try:
                    performance = Evaluation(y_pred=pred1, y_true=true1)
                    AUC_list.append(performance['AUROC'])
                except ValueError:
                    pass
            AUC = np.min(AUC_list) if AUC_list else float('nan')
            return (loss, pred, AUC)
        else:
            return (loss, pred)

    def apply_noise(self, X, ATAC, E, y, node_mask, region_mask, t=None):
        bs = X.shape[0]
        T = self.forward_steps
        device = self.device
        t_int = torch.randint(0, int(T), (bs, 1), device=device)
        t_float = t_int.float() / T
        s_float = (t_int.float() + 1) / T
        t_float = t_float.clamp(0.0001, 1 - 0.0001)
        s_float = s_float.clamp(0.0001, 1 - 0.0001)
        E_1_label = torch.argmax(E, dim=-1)
        prob_E_t = p_xt_g_x1(E1=E_1_label, t=t_float, limit_dist=self.limit_dist)
        X_t = X.clone()
        ATAC_t = ATAC.clone()
        sampled_t = flow_matching_utils.sample_discrete_features(probE=prob_E_t, node_mask=node_mask)
        E_t = F.one_hot(sampled_t.E, num_classes=2)
        z_t = PlaceHolder(X=X_t, ATAC=ATAC_t, E=E_t, y=y).type_as(X_t).mask(node_mask, region_mask)
        noisy_data = {'t': t_float, 'X_t': z_t.X, 'ATAC_t': z_t.ATAC, 'E_t': z_t.E, 'y_t': z_t.y, 's': s_float, 'node_mask': node_mask, 'region_mask': region_mask}
        return noisy_data

    def compute_step_probs(self, R_t, current_label, dt):
        raw_jump_probs = (R_t * dt).clamp(min=0.0)
        probs_jump = raw_jump_probs * (1.0 - current_label)
        jump_mass = probs_jump.sum(dim=-1, keepdim=True).clamp_max(1.0)
        stay_mass = 1.0 - jump_mass
        probs = probs_jump + stay_mass * current_label
        probs = probs.clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def forward(self, data, epoch, *args, **kwargs):

        adj_matrix, gene_exp, atac_exp, rtf, rg, node_mask, region_mask, g_emb, p_emb = (
            value.to(self.device) if value is not None else None for value in data)
        dense_data = PlaceHolder(X=gene_exp, ATAC=atac_exp, E=F.one_hot(adj_matrix.long(), num_classes=2).float(), y=None)
        dense_data = dense_data.mask(node_mask, region_mask)
        X, ATAC, E = (dense_data.X, dense_data.ATAC, dense_data.E)
        y = torch.empty(X.shape[0], 0)
        noisy_data = self.apply_noise(X, ATAC, E, y, node_mask, region_mask)
        noisy_data['rtf'] = rtf
        noisy_data['rg'] = rg
        noisy_data['g_emb'] = g_emb
        noisy_data['p_emb'] = p_emb
        loss, pred, AUC = self.p_losses(E, noisy_data, node_mask)
        return (loss, AUC)

    @torch.no_grad()
    def sample_zt(self, noisy_data):
        node_mask = noisy_data['node_mask']
        X_t = noisy_data['X_t']
        y_t = noisy_data['y_t']
        ATAC_t = noisy_data['ATAC_t']
        E_t = noisy_data['E_t']
        dt = noisy_data['dt']
        region_mask = noisy_data['region_mask']
        limit_e = self.limit_dist.E
        torch.set_grad_enabled(False)
        pred = self.model(node_mask, noisy_data)
        pred = self.pred_shape2data(pred, noisy_data)
        pred1 = pred.mask(node_mask)
        pred_E = F.softmax(pred1.E, dim=-1)
        R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(y_t, node_mask, E_t, pred_E)
        prob_E = self.compute_step_probs(R_t_E, E_t, dt)
        if y_t.flatten()[0].item() == 1.0 - 1e-05:
            prob_E = pred_E
        sampled_s = flow_matching_utils.sample_discrete_features(prob_E, node_mask=node_mask)
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()
        y_to_save = y_t
        out_one_hot = PlaceHolder(X=X_t, ATAC=ATAC_t, E=E_s, y=y_to_save)
        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        return (out_one_hot, pred_E)

    @torch.no_grad()
    def test_step(self, testdata, ensemble):
        from GenLink import evaluation
        metrics = ['AUROC', 'AUPRC', 'F1-score', 'precision']
        re_adj_all = []
        all_performances = []
        for idx, data in enumerate(testdata):
            print(float(idx) / len(testdata))
            pred_adj_all = []
            for i in range(ensemble):
                pred_adj, all_adj_list, adj_true, case_E = self.one_batch(data)
                pred_adj_all.append(case_E[:, :, 1])
            adj_final = cal_final_net2(pred_adj_all)
            performance = evaluation(self, adj_final, adj_true)
            all_performances.append(performance)
            re_adj_all.append(adj_final)
        print('\n--- Final Average Performance ---')
        avg_performance = {}
        for m in metrics:
            avg_val = np.mean([p[m] for p in all_performances])
            avg_performance[m] = avg_val
            print(f'Average {m}: {avg_val:.4f}')
        return (re_adj_all, avg_performance, case_E)

    @torch.no_grad()
    def one_batch(self, batch, batch_size=1):
        all_adj = []
        b_adj, b_x, b_atac, b_rtf, b_rg, b_mask, b_region_mask, g_emb, p_emb = (
            value.to(self.device) if value is not None else None for value in batch)
        z_T = flowmatching_utils.sample_discrete_feature_noise(X=b_x, ATAC=b_atac, limit_dist=self.limit_dist, node_mask=b_mask, region_mask=b_region_mask)
        b_E = z_T.E
        b_adj = b_adj.squeeze(0)
        b_adj = b_adj[:, b_mask.squeeze(0)]
        b_adj = b_adj[b_mask.squeeze(0), :]
        y = None
        N = self.backward_step + 1
        N_coarse = int(N * 0.3)
        N_fine = N - N_coarse
        ts_coarse = torch.linspace(0.0, 0.7, N_coarse + 1, device=self.device)
        ts_fine = torch.linspace(0.7, 1.0, N_fine + 1, device=self.device)
        ts = torch.cat([ts_coarse, ts_fine[1:]])
        show = True
        if show:
            pbar2 = tqdm(range(0, self.backward_step), ncols=100)
        else:
            pbar2 = range(0, self.backward_step)
        for t_int in pbar2:
            t_scalar = ts[t_int]
            s_scalar = ts[t_int + 1]
            dt = float((s_scalar - t_scalar).item())
            t = t_scalar.expand(batch_size, 1)
            s = s_scalar.expand(batch_size, 1)
            noisy_data = {'X_t': b_x, 'trueE': b_adj, 'E_t': b_E, 'y_t': y, 's': s, 't': t, 'dt': dt, 'ATAC_t': b_atac, 'node_mask': b_mask, 'rtf': b_rtf, 'rg': b_rg, 'region_mask': b_region_mask, 'g_emb': g_emb, 'p_emb': p_emb}
            noisy_data['y_t'] = noisy_data['t'].float()
            sampled_s, prob_E = self.sample_zt(noisy_data)
            b_E, y = (sampled_s.E, sampled_s.y)
            sampled_s1 = copy.deepcopy(sampled_s).mask(b_mask, collapse=True)
            sampled_s1.E = sampled_s1.E.squeeze(0)
            sampled_s1.E = sampled_s1.E[:, b_mask.squeeze(0)]
            sampled_s1.E = sampled_s1.E[b_mask.squeeze(0), :]
            eye = torch.eye(sampled_s1.E.shape[1], dtype=torch.bool).to(self.device)
            diag_mask = ~eye
            sampled_s1.E = sampled_s1.E * diag_mask
            if show:
                performance = Evaluation(y_pred=torch.flatten(sampled_s1.E), y_true=torch.flatten(b_adj))
                pbar2.set_description(f" t = {int((t[0][0] * self.backward_step).cpu().numpy()):3.0f} -- AUROC:  {performance['AUROC']:.4f} -- AUPRC:  {performance['AUPRC']:.4f} -- F1-score:  {performance['F1-score']:.4f} -- precision:  {performance['precision']:.4f}")
            adj = sampled_s1.E.cpu().detach().numpy().copy()
            if y is not None:
                adj = pd.DataFrame(adj)
            all_adj.append(adj)
        E_prob = prob_E.squeeze(0)
        E_prob = E_prob[:, b_mask.squeeze(0), :]
        E_prob = E_prob[b_mask.squeeze(0), :, :]
        E_prob = E_prob * diag_mask.unsqueeze(-1)
        return (sampled_s1.E, all_adj, b_adj, E_prob)
