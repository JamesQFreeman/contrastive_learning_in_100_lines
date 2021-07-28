from typing import ValuesView
import torch
from torch import nn

from augmentation import SimCLR_augment


def get_encoder(net: nn.Module) -> nn.Module:
    """ input a network and output it's convolutional feature encoder"""
    return nn.Sequential(*(list(net.children())[:-1]))


def MLP(in_size, out_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, out_size)
    )

def InfoNCE(x1,x2):


class SimCLR(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        num_features = net.fc.in_features
        self.augment1 = SimCLR_augment
        self.augment2 = SimCLR_augment

        self.encoder = get_encoder(net)
        self.projector = MLP(in_size=num_features, out_size=256)

        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        view1, view2 = self.augment1(x), self.augment2(x)
        proj1, proj2 = self.projector(self.encoder(
            view1)), self.projector(self.encoder(view2))
        pred1, pred2 = self.predictor(proj1), self.predictor(proj2)

        loss = nn
        loss = -(self.criterion(proj1, pred2).mean() +
                 self.criterion(proj2, pred1).mean()) * 0.5
        return loss.mean()
