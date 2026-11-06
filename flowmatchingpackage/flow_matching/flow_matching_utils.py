import torch

class PlaceHolder:

    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

def sample_discrete_features(probE, node_mask, mask=False):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _, _ = probE.shape
    probE = probE.reshape(bs * n * n, -1)
    if torch.isnan(probE).any():
        print('!!! 检测到 NaN 值 !!!')
        print(f'probE Max: {probE.max()}, Min: {probE.min()}')
    elif (probE < 0).any():
        print('!!! 检测到负数值 (可能是 Logits 未归一化) !!!')
        print(f'probE Min: {probE.min()}')
    elif probE.sum(dim=-1).min() <= 0:
        print('!!! 检测到总和为 0 的行 !!!')
        print(f'Min Sum: {probE.sum(dim=-1).min()}')
    E_t = probE.multinomial(1, replacement=True).reshape(bs, n, n)
    if mask:
        E_t = E_t * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    return PlaceHolder(X=None, E=E_t, y=torch.zeros(bs, 0).type_as(E_t))
