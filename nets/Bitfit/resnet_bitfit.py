from nets.Bitfit.resnet import resnet50


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



class ResNetBitFit(nn.Module):
    """
    embed_dim:96,128,192
    """
    def __init__(self, num_classes=21, pretrained=False, backbone="resnet50", base_size=64):
        super(ResNetBitFit, self).__init__()

        filters = [base_size, base_size * 2, base_size * 4, base_size * 8, base_size * 16]

        self.swin_backbone = resnet50(pretrained=pretrained)

        self.upSample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.Connection_Conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)
        )


        # 32,32,1536->64,64,512
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Conv_block5 = conv_block(filters[3] + 1024, filters[3])

        # 64,64,512->128,128,256
        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Conv_block4 = conv_block(filters[2] + 512, filters[2])

        # 128,128,256->256,256,128
        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Conv_block3 = conv_block(filters[1] + 256, filters[1])

        # 256,256,128->512,512,64
        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Conv_block2 = conv_block(filters[0] + 64, filters[0])

        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

        # Att之前需要保证shape和channel都一致
        self.upSample5 = up_conv(filters[4], filters[3], channel=1)
        self.upSample4 = up_conv(filters[3], filters[2], channel=1)
        self.upSample3 = up_conv(filters[2], filters[1], channel=1)
        self.upSample2 = nn.UpsamplingBilinear2d(scale_factor=2)

        # concat之前需要conv1x1
        self.convert1 = nn.Conv2d(512, 1024, kernel_size=1, padding=0)
        self.convert2 = nn.Conv2d(256, 512, kernel_size=1, padding=0)
        self.convert3 = nn.Conv2d(128, 256, kernel_size=1, padding=0)

        self.backbone = backbone
        self.frozen()

    def forward(self, inputs):

        [feat1, feat2, feat3, feat4, feat5] = self.swin_backbone.forward(inputs)

        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)
        # print(feat4.shape)
        # print(feat5.shape)

        connection_temp = self.Connection_Conv(feat5)
        d5 = self.Up5(connection_temp)
        e4 = self.Att5(g=d5, x=self.upSample5(feat4))
        e4 = self.convert1(e4)
        d5 = torch.cat((e4, d5), dim=1)
        a4 = self.Conv_block5(d5)
        # print(a4.shape)

        d4 = self.Up4(a4)
        e3 = self.Att4(g=d4, x=self.upSample4(feat3))
        e3 = self.convert2(e3)
        d4 = torch.cat((e3, d4), dim=1)
        a3 = self.Conv_block4(d4)
        # print(a3.shape)

        d3 = self.Up3(a3)
        e2 = self.Att3(g=d3, x=self.upSample3(feat2))
        e2 = self.convert3(e2)
        d3 = torch.cat((e2, d3), dim=1)
        a2 = self.Conv_block3(d3)
        # print(a2.shape)

        d2 = self.Up2(a2)
        e1 = self.Att2(g=d2, x=self.upSample2(feat1))
        d2 = torch.cat((e1, d2), dim=1)
        a1 = self.Conv_block2(d2)
        # print(a1.shape)

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
    model = ResNetBitFit(num_classes=3 + 1, pretrained=False, backbone="swin_T_224")
    a, b, c, d, e = getModelSize(model)
    x = torch.randn((2, 3, 224, 224))
    out = model(x)
    y1, y2 = model.calculate_unfrozen_parameter_ratio()
    print(y2)
    print(out.shape)
    # print(model)