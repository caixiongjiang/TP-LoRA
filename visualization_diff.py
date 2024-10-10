import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math
import copy
import sys
import os

from utils.utils import cvtColor, resize_image, preprocess_input

from nets.TP_LoRA.tp_lora import TP_LoRA
from nets.LoRA.lora import LoRA
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
        return (model_output[self.category, :, :] * self.mask).sum()


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


def cam_process(model, layer_name, layers, kid_dir, dataset="Orange-Navel-5.3k", target_class="rotten", heat_map=False):

    target_layers = [layers[layer_name]]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(classNames[dataset])}

    class_category = sem_class_to_idx[target_class]
    class_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    class_mask_float = np.float32(class_mask == class_category)
    targets = [SemanticSegmentationTarget(class_category, class_mask_float)]

    heat_map = False  # False: heatmap True: heatmap + original image
    print(f"Use {args.method}!")
    if heat_map == True:
        print("Save heatmap!")
    else:
        print("Save cam image!")

    with methods[args.method](model=model,
                              target_layers=target_layers,
                              reshape_transform=reshape_transform) as cam:


        gray_scale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
        print(gray_scale_cam.shape)
        cam_image = show_cam_on_image(preprocess_input(np.array(image_data, np.float32)), gray_scale_cam,
                                      use_rgb=True, image_weight=0.6, heat_map=heat_map)
    cam_image = Image.fromarray(cam_image)
    if heat_map == True:
        cam_image.save(f"{res_dir}/diff/{kid_dir}/{image_name}/{layer_name}_{image_name}_heatmap.jpg")
        print(f"Save {image_name}'s heatmap success!")
    else:
        cam_image.save(f"{res_dir}/diff/{kid_dir}/{image_name}/{layer_name}_{image_name}_cam.jpg")
        print(f"Save {image_name}'s cam image success!")



def multi_cam_process(model, layer_name, layers, kid_dir, dataset="Orange-Navel-5.3k", target_class="rotten", heat_map=False):

    gray_scale_cam = torch.randn(224, 224)
    # 形状为 (224, 224)，数据类型为 float32，设备为 CPU
    gray_scale_cam_combine = torch.zeros_like(gray_scale_cam)
    for x in layers[layer_name]:
        target_layers = [x]
        sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(classNames[dataset])}

        class_category = sem_class_to_idx[target_class]
        class_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        class_mask_float = np.float32(class_mask == class_category)
        targets = [SemanticSegmentationTarget(class_category, class_mask_float)]

        heat_map = False  # False: heatmap True: heatmap + original image
        print(f"Use {args.method}!")
        if heat_map == True:
            print("Save heatmap!")
        else:
            print("Save cam image!")

        with methods[args.method](model=model,
                                  target_layers=target_layers,
                                  reshape_transform=reshape_transform) as cam:
            gray_scale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
            print(gray_scale_cam.shape)
            gray_scale_cam_combine += gray_scale_cam

    cam_image = show_cam_on_image(preprocess_input(np.array(image_data, np.float32)), gray_scale_cam_combine,
                                      use_rgb=True, image_weight=0.6, heat_map=heat_map)
    cam_image = Image.fromarray(cam_image)
    if heat_map == True:
        cam_image.save(f"{res_dir}/diff/{kid_dir}/{image_name}/{layer_name}_{image_name}_heatmap.jpg")
        print(f"Save {image_name}'s heatmap success!")
    else:
        cam_image.save(f"{res_dir}/diff/{kid_dir}/{image_name}/{layer_name}_{image_name}_cam.jpg")
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

    # TP-LoRA模型
    model_path_tp_lora = "logs/Orange-Navel-5.3k/ablation/EP2-3&2-4/ep250-loss0.020-val_loss0.035.pth"
    Update_TP_LoRA_Set(mlp_dim=0.25, lora_dim=8, act="ReLU", in_location="ATT+MLP", out_location="ALL")
    model_tp_lora = TP_LoRA(text_size='TINY', dataset='Orange-Navel', num_classes=5 + 1, backbone='swin_T_224')

    model_tp_lora.load_state_dict(torch.load(model_path_tp_lora, map_location='cuda'))
    model_tp_lora.eval()

    # LoRA模型
    model_path_lora = "logs/Orange-Navel-5.3k/Swin-Tiny/LoRA/ep490-loss0.042-val_loss0.041.pth"
    model_lora = LoRA(num_classes=5 + 1, backbone='swin_T_224')

    model_lora.load_state_dict(torch.load(model_path_lora, map_location='cuda'))
    model_lora.eval()

    # 全局参数
    image_path = 'datasets/Orange-Navel-5.3k/VOC2007/JPEGImages/2021-06-08-143440-20_027.jpg'
    image = Image.open(image_path)
    image_name = image_path.split("/")[-1].split(".")[0]
    image = cvtColor(image)
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]
    image_data, nw, nh = resize_image(image, (224, 224))
    image_data_1 = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    image_tensor = torch.from_numpy(image_data_1)

    if args.use_cuda:
        model_tp_lora = model_tp_lora.cuda()
        model_lora = model_lora.cuda()
        image_tensor = image_tensor.cuda()

    # LoRA
    res_dir = "vis_res"
    image_dir_lora = res_dir + "/diff/lora" + f"/{image_name}"
    make_path(image_dir_lora)

    with open(f"{res_dir}/model_structure_lora.txt", 'w') as f:
        sys.stdout = f
        print(model_lora)
        sys.stdout = sys.__stdout__

    output = model_lora(image_tensor)
    normalized_masks = F.softmax(output, dim=1).cpu()
    classNames = {
        "Orange-Navel-5.3k": ['background', 'rotten', 'navel deformation', 'mild pitting', 'severe pitting',
                              'severe oil spotting'],
        "Lemon-2.7k": ["background", 'flaw', 'navel', 'illness', 'gangrene', 'decay'],
        "Grapefruit-1.9k": ["background", "stem", "defect", "navel"]
    }

    layers_lora = {
        "Backbone_Adapter_1_ATT_LoRA": model_lora.swin_backbone.layers[0].blocks[-1].attn.lora_v.adapter_up,
        "Backbone_Adapter_2_ATT_LoRA": model_lora.swin_backbone.layers[1].blocks[-1].attn.lora_v.adapter_up,
        "Backbone_Adapter_3_ATT_LoRA": model_lora.swin_backbone.layers[2].blocks[-1].attn.lora_v.adapter_up,
        "Adapter_CNN_1_LoRA": model_lora.lora_adapter1.adapter_up,
        "Adapter_CNN_2_LoRA": model_lora.lora_adapter2.adapter_up,
        "Adapter_CNN_3_LoRA": model_lora.lora_adapter3.adapter_up,
        "Adapter_CNN_4_LoRA": model_lora.lora_adapter4.adapter_up,
        "Adapter_ATT_1_LoRA": model_lora.Att1.lora_adapter_att.adapter_up,
        "Adapter_ATT_2_LoRA": model_lora.Att2.lora_adapter_att.adapter_up,
        "Adapter_ATT_3_LoRA": model_lora.Att3.lora_adapter_att.adapter_up,
        "Adapter_ATT_4_LoRA": model_lora.Att4.lora_adapter_att.adapter_up
    }

    for layer_name, _ in layers_lora.items():
        # cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="severe oil spotting")
        # cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="rotten")
        cam_process(model_lora, layer_name, layers_lora, "lora", dataset="Orange-Navel-5.3k", target_class="mild pitting")

    # TP-LoRA
    res_dir = "vis_res"
    image_dir_tp_lora = res_dir + "/diff/tp_lora" + f"/{image_name}"
    make_path(image_dir_tp_lora)

    with open(f"{res_dir}/model_structure_tp_lora.txt", 'w') as f:
        sys.stdout = f
        print(model_tp_lora)
        sys.stdout = sys.__stdout__

    output = model_tp_lora(image_tensor)
    normalized_masks = F.softmax(output, dim=1).cpu()
    classNames = {
        "Orange-Navel-5.3k": ['background', 'rotten', 'navel deformation', 'mild pitting', 'severe pitting',
                              'severe oil spotting'],
        "Lemon-2.7k": ["background", 'flaw', 'navel', 'illness', 'gangrene', 'decay'],
        "Grapefruit-1.9k": ["background", "stem", "defect", "navel"]
    }

    layers_tp_lora = {
        "Adapter_CNN_1_TP_LoRA": [model_tp_lora.tp_lora_adapter1.adapter_up, model_tp_lora.tp_lora_adapter1.mlp],
        "Adapter_CNN_2_TP_LoRA": [model_tp_lora.tp_lora_adapter2.adapter_up, model_tp_lora.tp_lora_adapter2.mlp],
        "Adapter_CNN_3_TP_LoRA": [model_tp_lora.tp_lora_adapter3.adapter_up, model_tp_lora.tp_lora_adapter3.mlp],
        "Adapter_CNN_4_TP_LoRA": [model_tp_lora.tp_lora_adapter4.adapter_up, model_tp_lora.tp_lora_adapter4.mlp],
        "Adapter_ATT_1_TP_LoRA": [model_tp_lora.Att1.tp_lora_adapter_cnn.adapter_up, model_tp_lora.Att1.tp_lora_adapter_cnn.mlp],
        "Adapter_ATT_2_TP_LoRA": [model_tp_lora.Att2.tp_lora_adapter_cnn.adapter_up, model_tp_lora.Att2.tp_lora_adapter_cnn.mlp],
        "Adapter_ATT_3_TP_LoRA": [model_tp_lora.Att3.tp_lora_adapter_cnn.adapter_up, model_tp_lora.Att3.tp_lora_adapter_cnn.mlp],
        "Adapter_ATT_4_TP_LoRA": [model_tp_lora.Att4.tp_lora_adapter_cnn.adapter_up, model_tp_lora.Att4.tp_lora_adapter_cnn.mlp],
        "Backbone_Adapter_1_ATT_TP_LoRA": [model_tp_lora.swin_backbone.layers[0].blocks[-1].attn.tp_lora_v.adapter_up,
                                       model_tp_lora.swin_backbone.layers[0].blocks[-1].attn.tp_lora_v.mlp],
        "Backbone_Adapter_2_ATT_TP_LoRA": [model_tp_lora.swin_backbone.layers[1].blocks[-1].attn.tp_lora_v.adapter_up,
                                           model_tp_lora.swin_backbone.layers[1].blocks[-1].attn.tp_lora_v.mlp],
        "Backbone_Adapter_3_ATT_TP_LoRA": [model_tp_lora.swin_backbone.layers[2].blocks[-1].attn.tp_lora_v.adapter_up,
                                           model_tp_lora.swin_backbone.layers[2].blocks[-1].attn.tp_lora_v.mlp]
    }

    for layer_name, _ in layers_tp_lora.items():
        # cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="severe oil spotting")
        # cam_process(layer_name, dataset="Orange-Navel-5.3k", target_class="rotten")
        multi_cam_process(model_tp_lora, layer_name, layers_tp_lora, "tp_lora", dataset="Orange-Navel-5.3k", target_class="mild pitting")






