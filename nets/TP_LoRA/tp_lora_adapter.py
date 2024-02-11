import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

from nets.TP_LoRA.utils import read_config, pad_sequence



def get_tp_lora_config():
    cfg = read_config()
    mlp_hidden_ratio = cfg['MODEL']['MLP_HIDDEN_RATIO']
    lora_dim = cfg['MODEL']['LORA_DIM']
    act = cfg['MODEL']['ACT']

    return act, lora_dim, mlp_hidden_ratio




class TP_LoRA_Adapter(nn.Module):
    def __init__(self, in_dim, text_vector, bit=32, size='TINY', dataset='Orange-Navel'):
        super().__init__()

        ACT, dim, mlp_hidden_ratio = get_tp_lora_config()
        self.words_vector = text_vector # [B, L, C]


        if bit == 32:
            self.adapter_down = nn.Linear(in_dim, dim, bias=False)
            self.adapter_up = nn.Linear(dim, in_dim, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            self.adapter_down = QLinear(in_dim, dim, bit)
            self.adapter_up = QLinear(dim, in_dim, bit)
            nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        if ACT == 'LoRA':
            self.act = nn.Identity()
        elif ACT == 'ReLU':
            self.act = nn.ReLU()
        elif ACT == 'GeLU':
            self.act = QuickGELU()
        else:
            raise ValueError("Unsupported 'ACT' value")
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(self.words_vector.shape[-1], int(mlp_hidden_ratio * dim)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_ratio * dim), in_dim)
        )

    def forward(self, x):
        B, L, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout1(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        if L < self.words_vector.shape[1]:
            x_pt = self.dropout2(self.mlp(self.words_vector[:, :L, :].to('cuda')))
        else:
            x_pt = self.dropout2(self.mlp(pad_sequence(self.words_vector, L).to('cuda')))

        return x_up + x_pt


class TP_LoRA_Adapter_CNN(nn.Module):
    def __init__(self, in_dim, text_vector, bit=32, size='TINY', dataset='Orange-Navel'):
        super().__init__()

        ACT, dim, mlp_hidden_ratio = get_tp_lora_config()
        self.words_vector = text_vector  # [B, L, C]

        if bit == 32:
            self.adapter_down = nn.Linear(in_dim, dim, bias=False)
            self.adapter_up = nn.Linear(dim, in_dim, bias=False)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            self.adapter_down = QLinear(in_dim, dim, bit)
            self.adapter_up = QLinear(dim, in_dim, bit)
            nn.init.trunc_normal_(self.adapter_up.weight, mean=0.0, std=0.001, a=-0.002, b=0.002)
        if ACT == 'LoRA':
            self.act = nn.Identity()
        elif ACT == 'ReLU':
            self.act = nn.ReLU()
        elif ACT == 'GeLU':
            self.act = QuickGELU()
        else:
            raise ValueError("Unsupported 'ACT' value")
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(self.words_vector.shape[-1], int(mlp_hidden_ratio * dim)),
            nn.ReLU(),
            nn.Linear(int(mlp_hidden_ratio * dim), in_dim)
        )


    def forward(self, x):
        # B, C, H, W -> B, H, W, C -> B, N, C
        x1 = x.transpose(1, 2).transpose(2, 3).contiguous().view(x.size(0), -1, x.size(1))
        B, L, C = x1.shape
        x_down = self.adapter_down(x1)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout1(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        if L < self.words_vector.shape[1]:
            x_pt = self.dropout2(self.mlp(self.words_vector[:, :L, :].to('cuda')))
        else:
            x_pt = self.dropout2(self.mlp(pad_sequence(self.words_vector, L).to('cuda')))

        x_up = x_up + x_pt
        x_up = x_up.view(x.size(0), x.size(2), x.size(3), x.size(1))
        # B, H, W, C -> B, C, H, W
        x_up = x_up.transpose(1, 3).transpose(2, 3).contiguous()

        return x_up


class QLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bits=1):
        super(QLinear, self).__init__(in_channels, out_channels, bias=False)
        self.q = self.quantize(bits)
        self.fake_quan = True
        self.bits = bits

    def forward(self, inputs):
        if not self.fake_quan:
            output = F.linear(inputs, self.weight, None)
            return output

        w = self.weight
        mean_w = w.mean()
        std_w = w.std()
        w = (w - mean_w) / (std_w + 1e-5)
        wb = self.q(w).data + w - w.data
        weight = wb * std_w + mean_w
        output = F.linear(inputs, weight, None)
        return output

    def dump(self):
        w = self.weight
        mean_w = w.mean()
        std_w = w.std()
        w = (w - mean_w) / (std_w + 1e-5)
        quantized = self.q(w).data.reshape(-1, 1)
        quantized = (quantized - self.code.reshape(1, -1)).abs().argmin(dim=1)
        byte_str = b''
        byte = 0
        for i in range(len(quantized)):
            if i % (8 // self.bits) == 0:
                byte = 0
            byte += quantized[i].item() * (2 ** self.bits) ** (i % (8 // self.bits))
            if i % (8 // self.bits) == (8 // self.bits) - 1:
                byte_str += byte.to_bytes(1, 'big')
        return byte_str, mean_w, std_w

    def load(self, byte_str, mean_w, std_w):
        self.fake_quan = False
        quantized = torch.zeros_like(self.weight.reshape(-1))
        for i in range(len(byte_str)):
            byte = byte_str[i]
            for j in range(8 // self.bits):
                quantized[i * (8 // self.bits) + j].data += self.code[byte % (2 ** self.bits)].data
                byte //= (2 ** self.bits)
        quantized = quantized * std_w + mean_w
        self.weight.data = quantized.reshape(*self.weight.size())

    def quantize(self, bit=1):
        j = 2 ** bit * 2
        ppf = np.array([0 for _ in range(1, j)])
        values = ppf[::2]
        ranges = ppf[1::2]
        for i in range(500):
            ranges = (values[1:] + values[:-1]) / 2
            pv = norm.cdf(ranges)
            pv = np.insert(pv, 0, 0)
            pv = np.insert(pv, len(pv), 1)
            pv = (pv[1:] + pv[:-1]) / 2
            values = norm.ppf(pv)
        value = torch.tensor(values).float()
        self.code = value
        pos = torch.tensor(ranges).float()
        delta = (value[1:] - value[:-1]) / 2

        def func(x):
            pos_ = pos.to(x.device)
            delta_ = delta.to(x.device)
            x = x.unsqueeze(dim=-1)
            x = x - pos_
            x = torch.sign(x)
            x *= delta_
            return x.sum(-1)

        return func

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


