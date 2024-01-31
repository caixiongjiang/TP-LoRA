import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, in_dim = 32, mlp_ratio=0.25, short_cut=False):
        super(Adapter, self).__init__()
        hidden_dims = int(in_dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(in_dim, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, in_dim)
        self.short_cut = short_cut

    def forward(self, x):

        x1 = self.fc1(x)
        x1 = self.act(x1)
        out = self.fc2(x1)
        if self.short_cut:
            return out + x
        return out





class Adapter_CNN(nn.Module):
    def __init__(self, in_dim = 32, mlp_ratio=0.25):
        super(Adapter_CNN, self).__init__()
        hidden_dims = int(in_dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(in_dim, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, in_dim)

    def forward(self, x):
        # B, C, H, W -> B, H, W, C -> B, N, C
        x1 = x.transpose(1, 2).transpose(2, 3).contiguous().view(x.size(0), -1, x.size(1))
        x1 = self.fc1(x1)
        x1 = self.act(x1)
        out = self.fc2(x1)
        out = out.view(x.size(0), x.size(2), x.size(3), x.size(1))
        # B, H, W, C -> B, C, H, W
        out = out.transpose(1, 3).transpose(2, 3).contiguous()

        return out + x