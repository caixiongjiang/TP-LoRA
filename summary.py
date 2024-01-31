'''
pip install thop
'''
import torch
from thop import profile

from nets.BaseModel.swinTS_Att_Unet import swinTS_Att_Unet
from nets.Bitfit.bitfit import BitFit
from nets.Adapter.adapter_tuning import Adapter_Tuning
from nets.ConvPass.convpass_tuning import ConvPass_Tuning
from nets.LoRA.lora import LoRA
from nets.AdaptFormer.adaptformer import AdaptFormer
from nets.VPT.vpt import VPT
from nets.TP_LoRA.tp_lora import TP_LoRA
from nets.TP_LoRA.utils import Update_TP_LoRA_Set

# TODO:增加消融实验设置的summary


def calculate(net, model_name, baseline=False):
    input = torch.randn(1, 3, 224, 224).to(device)
    net = net.to(device)
    if not baseline:
        unfrozen_param, ratio = net.calculate_unfrozen_parameter_ratio()
    flops, params = profile(net, inputs=(input, ))
    if not baseline:
        print('==========================')
        print(f'{model_name}:')
        print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
        print("params=", str(params/1e6)+'{}'.format("M"))
        print("Unfrozen params=", str(unfrozen_param/1e6)+'{}'.format("M"))
        print("Unfrozen Ratio=", str(round(ratio * 100, 2)) + '%')
        print('==========================')
    else:
        print('==========================')
        print(f"Baseline Model:{model_name}")
        print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
        print("params=", str(params / 1e6) + '{}'.format("M"))
        print('==========================')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("====================Backbone = Swin-Tiny=================")

    baseline_t = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(baseline_t, model_name='Swin-T-Att-UNet', baseline=True)

    baseline_t_class = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    baseline_t_class.frozen()
    calculate(baseline_t_class, model_name='Swin-T-Att-UNet', baseline=False)

    bitfit_t = BitFit(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(bitfit_t, model_name='BitFit')

    adapter_tuning_t = Adapter_Tuning(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(adapter_tuning_t, model_name='Adapter-Tuning')

    convpass_tuning_t = ConvPass_Tuning(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(convpass_tuning_t, model_name='ConvPass')

    lora_t = LoRA(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(lora_t, model_name='LoRA')

    adaptformer_t = AdaptFormer(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(adaptformer_t, model_name='AdaptFormer')

    vpt_t = VPT(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(vpt_t, model_name='VPT')

    tp_lora_t = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(tp_lora_t, model_name='TP-LoRA')

    print("====================Backbone = Swin-Small=================")

    baseline_s = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(baseline_s, model_name='Swin-S-Att-UNet', baseline=True)

    baseline_s_class = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    baseline_s_class.frozen()
    calculate(baseline_s_class, model_name='Swin-S-Att-UNet', baseline=False)

    bitfit_s = BitFit(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(bitfit_s, model_name='BitFit')

    adapter_tuning_s = Adapter_Tuning(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(adapter_tuning_s, model_name='Adapter-Tuning')

    convpass_tuning_s = ConvPass_Tuning(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(convpass_tuning_s, model_name='ConvPass')

    lora_s = LoRA(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(lora_s, model_name='LoRA')

    adaptformer_s = AdaptFormer(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(adaptformer_s, model_name='AdaptFormer')

    vpt_s = VPT(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(vpt_s, model_name='VPT')

    tp_lora_s = TP_LoRA(text_size='TINY', dataset='Orange-Navel', num_classes=3 + 1, pretrained=False, backbone="swin_S_224")
    calculate(tp_lora_s, model_name='TP-LoRA')


