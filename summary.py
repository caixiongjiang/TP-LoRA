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


def calculate(net, model_name, basemodel=False):
    input = torch.randn(1, 3, 224, 224).to(device)
    net = net.to(device)
    if not basemodel:
        unfrozen_param, ratio = net.calculate_unfrozen_parameter_ratio()
    flops, params = profile(net, inputs=(input, ))
    if not basemodel:
        print('==========================')
        print(f'{model_name}:')
        print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
        print("params=", str(params/1e6)+'{}'.format("M"))
        print("Unfrozen params=", str(unfrozen_param/1e6)+'{}'.format("M"))
        print("Unfrozen Ratio=", str(round(ratio * 100, 2)) + '%')
        print('==========================')
    else:
        print('==========================')
        print(f"basemodel Model:{model_name}")
        print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
        print("params=", str(params / 1e6) + '{}'.format("M"))
        print('==========================')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=====================Ablation Study======================")
    print("=========================================================")
    print("=====================Experinment 1======================")

    Update_TP_LoRA_Set(mlp_dim=0.125, lora_dim=8, act='LoRA', in_location='ATT', out_location='ALL')
    model_1_1 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_1_1, model_name='Ablation model 1-1', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.125, lora_dim=8, act='GeLU', in_location='ATT', out_location='ALL')
    model_1_2 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_1_2, model_name='Ablation model 1-2', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.125, lora_dim=8, act='ReLU', in_location='ATT', out_location='ALL')
    model_1_3 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_1_3, model_name='Ablation model 1-3', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT', out_location='ALL')
    model_1_4 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_1_4, model_name='Ablation model 1-4', basemodel=False)

    print("=====================Experinment 2======================")

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT', out_location='ALL')
    model_2_1 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_2_1, model_name='Ablation model 2-1', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='MLP', out_location='ALL')
    model_2_2 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_2_2, model_name='Ablation model 2-2', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='ALL')
    model_2_3 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_2_3, model_name='Ablation model 2-3', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='ALL')
    model_2_4 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_2_4, model_name='Ablation model 2-4', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='DEEP')
    model_2_5 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_2_5, model_name='Ablation model 2-5', basemodel=False)

    print("=====================Experinment 3======================")

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='DEEP')
    model_3_1 = TP_LoRA(text_size='TINY', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_3_1, model_name='Ablation model 3-1', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='DEEP')
    model_3_2 = TP_LoRA(text_size='BASE', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_3_2, model_name='Ablation model 3-2', basemodel=False)

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='DEEP')
    model_3_3 = TP_LoRA(text_size='LARGE', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(model_3_3, model_name='Ablation model 3-3', basemodel=False)
   
    print("====================Backbone = Swin-Tiny=================")

    basemodel_t = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(basemodel_t, model_name='Swin-T-Att-UNet', basemodel=True)

    basemodel_t_class = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    basemodel_t_class.frozen()
    calculate(basemodel_t_class, model_name='Swin-T-Att-UNet', basemodel=False)

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

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='DEEP')
    tp_lora_t = TP_LoRA(text_size='LARGE', dataset='Orange-Navel',num_classes=5 + 1, pretrained=False, backbone="swin_T_224")
    calculate(tp_lora_t, model_name='TP-LoRA')

    print("====================Backbone = Swin-Small=================")

    basemodel_s = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    calculate(basemodel_s, model_name='Swin-S-Att-UNet', basemodel=True)

    basemodel_s_class = swinTS_Att_Unet(num_classes=5 + 1, pretrained=False, backbone="swin_S_224")
    basemodel_s_class.frozen()
    calculate(basemodel_s_class, model_name='Swin-S-Att-UNet', basemodel=False)

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

    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act='ReLU', in_location='ATT+MLP', out_location='DEEP')
    tp_lora_s = TP_LoRA(text_size='LARGE', dataset='Orange-Navel', num_classes=3 + 1, pretrained=False, backbone="swin_S_224")
    calculate(tp_lora_s, model_name='TP-LoRA')


