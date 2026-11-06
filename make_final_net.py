import pandas as pd
import torch

def cal_final_net2(data, alpha=2.0):
    avg_graph = torch.stack(data).float().mean(dim=0)
    p = avg_graph.clamp(1e-06, 1 - 1e-06)
    calibrated = p ** alpha / (p ** alpha + (1 - p) ** alpha)
    return pd.DataFrame(calibrated.cpu().numpy())
