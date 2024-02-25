import argparse
from typing import Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math
import copy
import sys
import os
import cv2

from utils.utils import cvtColor, resize_image, preprocess_input

from nets.TP_LoRA.tp_lora import TP_LoRA
from nets.TP_LoRA.utils import Update_TP_LoRA_Set

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image



class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()
    
def reshape_transform(tensor):
    # input tensor: (B, H*W, C)
    result = tensor.reshape(tensor.size(0),
        int(math.sqrt(tensor.size(1))), int(math.sqrt(tensor.size(1))), tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2).contiguous()
    return result

def make_path(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass


def cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="rotten", heat_map=False):
    target_layers = [layers[layer_name]]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(classNames[dataset])}     

    class_category = sem_class_to_idx[target_class]
    class_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    class_mask_float = np.float32(class_mask == class_category)
    targets = [SemanticSegmentationTarget(class_category, class_mask_float)]  
    
    heat_map = False # False: heatmap True: heatmap + original image
    print(f"Use {args.method}!")
    if heat_map == True:
        print("Save heatmap!")
    else:
        print("Save cam image!")
    if layer_name == "final_conv":
        with methods[args.method](model=model,
                                target_layers=target_layers) as cam:
            gray_scale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
            gray_scale_cam = gray_scale_cam[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh), \
                    int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
            gray_scale_cam = cv2.resize(gray_scale_cam, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            cam_image = show_cam_on_image(preprocess_input(np.array(old_img, np.float32)), gray_scale_cam, use_rgb=True, image_weight=0.6, heat_map=heat_map)
            cam_image = cv2.resize(cam_image, (512, 512), interpolation = cv2.INTER_LINEAR)
        cam_image = Image.fromarray(cam_image)
        if heat_map == True:
            cam_image.save(f"{res_dir}/{image_name}/{layer_name}_{image_name}_heatmap.jpg")
            print(f"Save {image_name}'s {layer_name} heatmap success!")
        else:
            cam_image.save(f"{res_dir}/{image_name}/{layer_name}_{image_name}_cam.jpg")
            print(f"Save {image_name}'s cam image success!")
    else:
        with methods[args.method](model=model,
                                target_layers=target_layers,
                                reshape_transform=reshape_transform) as cam:
            gray_scale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
            # Remove the extra part of the gray bar.
            gray_scale_cam = gray_scale_cam[int((input_shape[0] - nh) // 2) : int((input_shape[0] - nh) // 2 + nh), \
                    int((input_shape[1] - nw) // 2) : int((input_shape[1] - nw) // 2 + nw)]
            gray_scale_cam = cv2.resize(gray_scale_cam, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            cam_image = show_cam_on_image(preprocess_input(np.array(old_img, np.float32)), gray_scale_cam, use_rgb=True, image_weight=0.6, heat_map=heat_map)
            # In order to facilitate a clearer display, save as 512*512
            cam_image = cv2.resize(cam_image, (512, 512), interpolation = cv2.INTER_LINEAR)
        cam_image = Image.fromarray(cam_image)
        if heat_map == True:
            cam_image.save(f"{res_dir}/{image_name}/{layer_name}_{image_name}_heatmap.jpg")
            print(f"Save {image_name}'s heatmap success!")
        else:
            cam_image.save(f"{res_dir}/{image_name}/{layer_name}_{image_name}_cam.jpg")
            print(f"Save {image_name}'s cam image success!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='layercam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args



if __name__ == '__main__':

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    
    model_path = "logs/Orange-Navel-5.3k/ablation/EP2-3&2-4/ep500-loss0.026-val_loss0.033.pth"
    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act="ReLU", in_location="ATT+MLP", out_location="ALL")
    model = TP_LoRA(text_size='TINY', dataset='Orange-Navel', num_classes=5 + 1, backbone='swin_T_224')
    # model = TP_LoRA(text_size='TINY', dataset='Lemon', num_classes=5 + 1, backbone='swin_T_224')
    # model = TP_LoRA(text_size='TINY', dataset='Grapefruit', num_classes=3 + 1, backbone='swin_T_224')

    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()

    # 全局参数
    # image_path = 'logs/images/2021-01-27-142436-3518_021.jpg'
    # image_path = 'logs/images/2021-05-20-084345-471_30.jpg'
    image_path = 'logs/images/2021-06-08-143440-20_027.jpg'
    image = Image.open(image_path)
    image_name = image_path.split("/")[-1].split(".")[0]
    image = cvtColor(image)
    old_img     = copy.deepcopy(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    input_shape = [224, 224]
    # If the dimensions do not match, add gray bars.
    image_data, nw, nh  = resize_image(image, (input_shape[1], input_shape[0])) 
    image_data_1  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    image_tensor = torch.from_numpy(image_data_1)

    if args.use_cuda:
        model = model.cuda()
        image_tensor = image_tensor.cuda()

    res_dir = "vis_res"
    image_dir = res_dir + f"/{image_name}"
    make_path(image_dir)
    with open(f"{res_dir}/model_structure.txt", 'w') as f:
        sys.stdout = f
        print(model)
        sys.stdout = sys.__stdout__


    output = model(image_tensor)
    normalized_masks = F.softmax(output, dim=1).cpu()
    classNames = {
        "Orange-Navel-5.3k": ['background', 'rotten', 'navel deformation', 'mild pitting', 'severe pitting', 'severe oil spotting'], 
        "Lemon-2.7k": ["background", 'flaw', 'navel', 'illness', 'gangrene', 'decay'], 
        "Grapefruit-1.9k": ["background", "stem", "defect", "navel"]
    }

    layers = {
        "final_conv": model.final_conv,
        "Adapter_CNN_1_LoRA": model.tp_lora_adapter1.adapter_up,
        "Adapter_CNN_2_LoRA": model.tp_lora_adapter2.adapter_up,
        "Adapter_CNN_3_LoRA": model.tp_lora_adapter3.adapter_up,
        "Adapter_CNN_4_LoRA": model.tp_lora_adapter4.adapter_up,
        "Adapter_CNN_1_Text": model.tp_lora_adapter1.mlp,
        "Adapter_CNN_2_Text": model.tp_lora_adapter2.mlp,
        "Adapter_CNN_3_Text": model.tp_lora_adapter3.mlp,
        "Adapter_CNN_4_Text": model.tp_lora_adapter4.mlp,
        "Adapter_ATT_1_LoRA": model.Att1.tp_lora_adapter_cnn.adapter_up,
        "Adapter_ATT_2_LoRA": model.Att2.tp_lora_adapter_cnn.adapter_up,
        "Adapter_ATT_3_LoRA": model.Att3.tp_lora_adapter_cnn.adapter_up,
        "Adapter_ATT_4_LoRA": model.Att4.tp_lora_adapter_cnn.adapter_up,
        "Adapter_ATT_1_Text": model.Att1.tp_lora_adapter_cnn.mlp,
        "Adapter_ATT_2_Text": model.Att2.tp_lora_adapter_cnn.mlp,
        "Adapter_ATT_3_Text": model.Att3.tp_lora_adapter_cnn.mlp,
        "Adapter_ATT_4_Text": model.Att4.tp_lora_adapter_cnn.mlp,
        "Backbone_Adapter_1_ATT_LoRA": model.swin_backbone.layers[0].blocks[-1].attn.tp_lora_v.adapter_up,
        "Backbone_Adapter_1_MLP_LoRA": model.swin_backbone.layers[0].blocks[-1].tp_lora_mlp.adapter_up,
        "Backbone_Adapter_2_ATT_LoRA": model.swin_backbone.layers[1].blocks[-1].attn.tp_lora_v.adapter_up,
        "Backbone_Adapter_2_MLP_LoRA": model.swin_backbone.layers[1].blocks[-1].tp_lora_mlp.adapter_up,
        "Backbone_Adapter_3_ATT_LoRA": model.swin_backbone.layers[2].blocks[-1].attn.tp_lora_v.adapter_up,
        "Backbone_Adapter_3_MLP_LoRA": model.swin_backbone.layers[2].blocks[-1].tp_lora_mlp.adapter_up,
        "Backbone_Adapter_1_Text": model.swin_backbone.layers[0].blocks[-1].tp_lora_mlp.mlp,
        "Backbone_Adapter_2_Text": model.swin_backbone.layers[1].blocks[-1].tp_lora_mlp.mlp,
        "Backbone_Adapter_3_Text": model.swin_backbone.layers[2].blocks[-1].tp_lora_mlp.mlp
    }

    for layer_name, _ in layers.items():
        # cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="severe oil spotting")
        # cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="rotten")
        cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="mild pitting")
        
