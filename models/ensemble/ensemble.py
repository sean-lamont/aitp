from models.get_model import get_model
import torch
import torch.nn as nn


class EnsembleEmbedder(nn.Module):
    def __init__(self, models, type='sum'):
        super(EnsembleEmbedder, self).__init__()
        self.models = [get_model(model) for model in models]
        self.type = type

    def forward(self, data):
        outs = torch.stack([model(data) for model in self.models], dim=0)

        if self.type == 'sum':
            return torch.sum(outs, dim=0)
        elif self.type == 'max':
            return torch.max(outs, dim=0)
        elif self.type == 'mean':
            return torch.mean(outs, dim=0)
        else:
            raise NotImplementedError
