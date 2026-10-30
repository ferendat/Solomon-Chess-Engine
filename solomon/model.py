import torch
import torch.nn as nn

from .config import MATERIAL_SCALE, MAX_CORRECTION


class ChessNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(768, hidden), nn.ReLU(),
            nn.Linear(hidden, 64), nn.ReLU(),
        )
        self.out = nn.Linear(64, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, material):
        correction = MAX_CORRECTION * torch.tanh(self.out(self.body(x)).squeeze(-1))
        return torch.tanh(material / MATERIAL_SCALE + correction)
