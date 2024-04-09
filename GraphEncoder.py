import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.drop(F.relu(self.linear(x)))
        return x


class GraphModel(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim):
        super(GraphModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.graphlayer = GraphLayer(input_dim, hidden_dim)

    def forward(self, x, mask):
        x = self.graphlayer(x)
        mask = mask[:, :, None].repeat(1, 1, self.hidden_dim)
        x = torch.sum(x * mask, dim=1)
        return x
