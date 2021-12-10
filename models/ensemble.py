import torch.nn as nn
import torch
class Ens(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_ens = len(models)
    def forward(self, x):
        out = []
        for model in self.models:
            out.append(model(x))
        return out