import torch
import torch.nn as nn

import math

class Convpass(nn.Module):
    def __init__(self, in_dim, dim=8, xavier_init=True):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(in_dim, dim)
        self.adapter_up = nn.Linear(dim, in_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        x_patch = self.adapter_conv(x_patch)
        x_down = x_patch.permute(0, 2, 3, 1).reshape(B, -1, self.dim)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up

class Convpass_CNN(nn.Module):
    def __init__(self, in_dim, dim=8, xavier_init=True):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Conv2d(in_dim, dim, 1, 1, 0)
        self.adapter_up = nn.Conv2d(dim, in_dim, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        x_down = self.adapter_down(x)
        x_patch = self.act(x_down)
        x_patch = self.adapter_conv(x_patch)
        x_down = self.act(x_patch)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)