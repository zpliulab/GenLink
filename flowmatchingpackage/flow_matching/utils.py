import torch.nn.functional as F

def p_xt_g_x1(E1, t, limit_dist):
    device = E1.device
    limit_dist.E = limit_dist.E.to(device)
    t_time = t.squeeze(-1)[:, None, None]
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()
    Et = t_time[:, None] * E1_onehot + (1 - t_time[:, None]) * limit_dist.E[None, None, None, :]
    return Et.clamp(min=0.0, max=1.0)

def dt_p_xt_g_x1(E1, limit_dist):
    device = E1.device
    limit_dist.E = limit_dist.E.to(device)
    E1_onehot = F.one_hot(E1, num_classes=len(limit_dist.E)).float()
    dE = E1_onehot - limit_dist.E[None, None, None, :]
    assert (dE.sum(-1).abs() < 0.0001).all()
    return dE
