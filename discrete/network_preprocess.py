import torch

class PlaceHolder:

    def __init__(self, X, E, y, ATAC=None, MAP=None):
        self.X = X
        self.E = E
        self.y = y
        self.ATAC = ATAC
        self.MAP = MAP

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.ATAC is not None:
            self.ATAC = self.ATAC.type_as(x)
        if self.MAP is not None:
            self.MAP = self.MAP.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, region_mask=None, collapse=False):
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        if collapse:
            self.E = torch.argmax(self.E, dim=-1)
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            if self.ATAC is not None and region_mask is not None:
                r_mask = region_mask.unsqueeze(-1)
                self.ATAC = self.ATAC * r_mask
            self.E = self.E * e_mask1 * e_mask2
        return self
