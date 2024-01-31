from nets.Bitfit.swin_transformer import swin_tiny_patch4_window7_224 as create_model_T_224
from nets.Bitfit.swin_transformer import swin_small_patch4_window7_224 as create_model_S_224


import torch.nn as nn
import torch



class small_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(small_conv_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block(x)

        return out



class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.conv(x)
        return out


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, channel=3):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        if channel == 1:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        out = self.up(x)
        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out



class BitFit(nn.Module):
    """
    embed_dim:96,128,192
    """
    def __init__(self, num_classes=21, pretrained=False, backbone="swin_T_224", embed_dim=96, base=64):
        super(BitFit, self).__init__()

        if backbone == "swin_T_224":
            self.swin_backbone = create_model_T_224(num_classes=1000)
        elif backbone == "swin_S_224":
            self.swin_backbone = create_model_S_224(num_classes=1000)

        # 将swin transfomer的分类头去掉
        remove_head = nn.Sequential()
        self.swin_backbone.avgpool = remove_head
        self.swin_backbone.head = remove_head


        filters = [base, base * 2, base * 4, base * 8, base * 16]

        # concat之前的通道转变
        self.convert1 = nn.Conv2d(filters[3], 2 * embed_dim, kernel_size=1, padding=0)
        self.convert2 = nn.Conv2d(filters[2], embed_dim, kernel_size=1, padding=0)
        self.convert3 = nn.Conv2d(filters[1], base*2, kernel_size=1, padding=0)
        self.convert4 = nn.Conv2d(filters[0], base, kernel_size=1, padding=0)


        # Attention block
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Att1 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)

        # bottle_neck
        self.bottle = nn.Sequential(
            nn.Conv2d(in_channels=4 * embed_dim, out_channels=filters[4], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )

        # 上采样+卷积
        self.up4 = up_conv(filters[4], filters[3])
        self.up3 = up_conv(filters[3], filters[2])
        self.up2 = up_conv(filters[2], filters[1])
        self.up1 = up_conv(filters[1], filters[0])

        # 卷积块
        self.conv4 = conv_block(2 * embed_dim + filters[3], filters[3])
        self.conv3 = conv_block(embed_dim + filters[2], filters[2])
        self.conv2 = conv_block(base*2 + filters[1], filters[1])
        self.conv1 = conv_block(base + filters[0], filters[0])

        self.final_conv = nn.Conv2d(in_channels=filters[0], out_channels=num_classes, kernel_size=1, padding=0)

        self.backbone = backbone
        self.embed_dim = embed_dim

        # 浅层与UNet保持一致
        self.cnn1 = conv_block(3, base)
        self.cnn2 = conv_block(base, 2 * base)
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Att之前需要改变通道数
        self.change1 = nn.Sequential(
            nn.Conv2d(embed_dim*2, filters[3], kernel_size=1, padding=0),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True)
        )
        self.change2 = nn.Sequential(
            nn.Conv2d(embed_dim, filters[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        # 锁层
        self.frozen()


    def forward(self, x):

        # 辅助网络捕获空间信息
        x1 = self.cnn1(x)

        x2 = self.downSample(x1)
        x2 = self.cnn2(x2)

        x, H, W, feat1, feat2, feat3 = self.swin_backbone(x)
        # print(x.shape)
        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)



        # 转回卷积网络所需要尺寸
        # x = x.view(-1, 8 * self.embed_dim, H, W)

        _, size1, C1 = feat1.shape
        feat1 = feat1.permute(0, 2, 1).contiguous().view(-1, C1, size1//(8*H), 8*H)

        _, size2, C2 = feat2.shape
        feat2 = feat2.permute(0, 2, 1).contiguous().view(-1, C2, size2//(4*H), 4*H)

        _, size3, C3 = feat3.shape
        feat3 = feat3.permute(0, 2, 1).contiguous().view(-1, C3, size3//(2*H), 2*H)

        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)
        # print(x.shape)

        # 中间连接层
        bottle = self.bottle(feat3)

        d4 = self.up4(bottle)
        e4 = self.Att4(g=d4, x=self.change1(feat2))
        e4 = self.convert1(e4)
        d4 = torch.concat([d4, e4], dim=1)
        a4 = self.conv4(d4)

        d3 = self.up3(a4)
        e3 = self.Att3(g=d3, x=self.change2(feat1))
        e3 = self.convert2(e3)
        d3 = torch.concat([d3, e3], dim=1)
        a3 = self.conv3(d3)

        d2 = self.up2(a3)
        e2 = self.Att2(g=d2, x=x2)
        e2 = self.convert3(e2)
        d2 = torch.concat([d2, e2], dim=1)
        a2 = self.conv2(d2)

        d1 = self.up1(a2)
        e1 = self.Att1(g=d1, x=x1)
        e1 = self.convert4(e1)
        d1 = torch.concat([d1, e1], dim=1)
        a1 = self.conv1(d1)
        out = self.final_conv(a1)

        return out

    def frozen(self):
        # 先锁住所有层
        for param in self.parameters():
            param.requires_grad = False

        # 开启所有Bias参数
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
        # 开启num_classes层的参数
        for param in self.final_conv.parameters():
            param.requires_grad = True

    def calculate_unfrozen_parameter_ratio(self):
        total_parameters = 0
        unfrozen_parameters = 0

        for param in self.parameters():
            total_parameters += param.numel()
            if param.requires_grad:
                unfrozen_parameters += param.numel()

        unfrozen_ratio = unfrozen_parameters / total_parameters
        return unfrozen_parameters, round(unfrozen_ratio, 4)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)




if __name__ == '__main__':
    model = BitFit(num_classes=3 + 1, pretrained=False, backbone="swin_T_224")
    a, b, c, d, e = getModelSize(model)
    x = torch.randn((2, 3, 224, 224))
    out = model(x)
    y1, y2 = model.calculate_unfrozen_parameter_ratio()
    print(y2)
    print(out.shape)
    # print(model)