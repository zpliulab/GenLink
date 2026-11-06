import torch
import torch.nn.functional as F
import flowmatchingpackage.flow_matching.flow_matching_utils as flow_matching_utils
from flowmatchingpackage.flow_matching.utils import dt_p_xt_g_x1, p_xt_g_x1

class RateMatrixDesigner:

    def __init__(self, limit_dist):
        self.limit_dist = limit_dist

    def compute_graph_rate_matrix(self, t, node_mask, G_t, G_1_pred):
        E_t = G_t
        E_1_pred = G_1_pred
        E_t_label = E_t.argmax(-1, keepdim=True)
        sampled_G_1 = flow_matching_utils.sample_discrete_features(E_1_pred, node_mask=node_mask)
        E_1_sampled = sampled_G_1.E
        dfm_variables = self.compute_dfm_variables(t, E_t_label, E_1_sampled)
        Rstar_t_E = self.compute_Rstar(dfm_variables)
        R_t_E = Rstar_t_E
        R_t_E = self.stabilize_rate_matrix(R_t_E, dfm_variables)
        return R_t_E

    def compute_dfm_variables(self, t, E_t_label, E_1_sampled):
        pt_vals_E = p_xt_g_x1(E_1_sampled, t, self.limit_dist)
        dt_p_vals_E = dt_p_xt_g_x1(E_1_sampled, self.limit_dist)
        pt_vals_at_Et = pt_vals_E.gather(-1, E_t_label).squeeze(-1)
        dt_p_vals_at_Et = dt_p_vals_E.gather(-1, E_t_label).squeeze(-1)
        Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)
        dfm_variables = {'pt_vals_E': pt_vals_E, 'pt_vals_at_Et': pt_vals_at_Et, 'dt_p_vals_E': dt_p_vals_E, 'dt_p_vals_at_Et': dt_p_vals_at_Et, 'Z_t_E': Z_t_E}
        return dfm_variables

    def compute_Rstar(self, dfm_variables):
        dt_p_vals_E = dfm_variables['dt_p_vals_E']
        dt_p_vals_at_Et = dfm_variables['dt_p_vals_at_Et']
        pt_vals_at_Et = dfm_variables['pt_vals_at_Et']
        Z_t_E = dfm_variables['Z_t_E']
        inner_E = dt_p_vals_E - dt_p_vals_at_Et[:, :, :, None]
        Rstar_t_numer_E = F.relu(inner_E)
        Rstar_t_denom_E = Z_t_E * pt_vals_at_Et
        Rstar_t_E = Rstar_t_numer_E / (Rstar_t_denom_E[:, :, :, None] + 1e-12)
        return Rstar_t_E

    def stabilize_rate_matrix(self, R_t_E, dfm_variables):
        pt_vals_E = dfm_variables['pt_vals_E']
        pt_vals_at_Et = dfm_variables['pt_vals_at_Et']
        R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_E[R_t_E > 100000.0] = 0.0
        de = R_t_E.shape[-1]
        R_t_E[(pt_vals_at_Et == 0.0)[:, :, :, None].repeat(1, 1, 1, de)] = 0.0
        R_t_E[pt_vals_E == 0.0] = 0.0
        return R_t_E
