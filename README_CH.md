## TP-LoRA: Parameter-Efﬁcient Fine-Tuning for citrus fruit defect segmentation model based on text prompt.

[English](README.md)

这是我们不久前工作的官方资料库：TP-LoRA([PDF]())

### 新闻
---

### 亮点
---

* 不同方法在柑橘类水果数据集上的性能对比其调整的参数量：

<p align="center">
<img src="Images/params_mIoU_fig.jpg" alt="Image 1" width="400"/>
</p>

### 更新
---
- [x] 完成代码和数据集的上传。（2024 1.30）


### 概述
---
* 柑橘类水果缺陷分割模型的参数高效微调（PEFT）工作流程：

![](Images/PEFT_process.jpg)


* 我们提出的基于文本提示的LoRA PEFT方法（TP-LoRA）的基本架构概览：

![](Images/TP-LoRA.jpg)

* 将TP-LoRA应用于Swin-Att-UNet基本模型的方案：

![](Images/BaseModel-TP-LoRA.jpg)

### 数据集
---
* 柑橘类水果缺陷数据集：

| Datasets | Image size | Quantities | Enhance | Usage | Link |
|:--------------------:|:----------------:|:-----------------:|:----------------:|:----------------:|:----------------:|
| Orange-Navel-4.5k | $512\times 512$ | 4344 | True| Pretrain|[download]()|
| Orange-Navel-5.3k | $144\times 144$ | 5290 | False | PEFT | × |
| Lemon-2.7k | $1024\times 1024$ | 2690 | False | PEFT | [download]() |
| Grapefruit-1.9k | $512\times 512$ | 1933 | False | PEFT | [download]() |

x: 出于商业原因，该数据集不公开。

*注意：我们仅在Orange-Navel-4.5k中提供1448个肚脐橙色缺陷原始数据集，如果您想扩展数据集，您可以使用[Imgaug for segmentation maps and mask](https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html)来增强数据。*

### 预训练模型
---
* 预训练的骨干网络：

| Model(Imagenet-1k) | Input size | ckpt |
|:--------------------:|:---------------------:|:---------------------:|
| Swin-Tiny | $224\times 224$ | [download]() |
| Swin-Small | $224\times 224$ | [download]() |

* 预训练的基本分割网络：

| Model(Orange-Navel-4.5k) | Input size | mIoU(%) | ckpt |
|:--------------------:|:---------------------:|:---------------------:|:---------------------:|
| Swin-T-Att-UNet | $224\times 224$ | 89.75 | [download]() |
| Swin-S-Att-UNet | $224\times 224$ | 89.92 | [download]() |

* 文本编码网络：

| Model | ckpt |
|:--------------------:|:---------------------:|
| Bert-base |  [download]() |

### 结果
---
* 柑橘数据集基准的全面性能（mIoU）：

<table>
    <tr>
	    <th colspan="8">Comparison of the effects of different PEFT methods.</th>
	</tr >
	<tr>
	    <td rowspan="2" style="text-align: center;">Method</td>
	    <td rowspan="2" style="text-align: center;">Backbone</td>
        <td rowspan="2" style="text-align: center;">Params(M)</td> 
      <td colspan="4" style="text-align: center;">mIoU(%)</td>
	</tr >
    <tr> 
      <td style="text-align: center;">ON-5.3k</td> 
      <td style="text-align: center;">L-2.7k</td>
      <td style="text-align: center;">G-1.9k</td>
      <td style="text-align: center;">Avg.</td>
	</tr >
    <tr>
	    <td style="text-align: center;">Full</td>
	    <td rowspan="9" style="text-align: center;">Swin-Tiny</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">Classify</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">BitFit</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">VPT</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">Adapter</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">AdaptFormer</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">LoRA</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">Convpass</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">TP-LoRA(ours)</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
	<tr>
	    <td style="text-align: center;">Full</td>
	    <td rowspan="9" style="text-align: center;">Swin-Small</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">Classify</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">BitFit</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">VPT</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">Adapter</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">AdaptFormer</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">LoRA</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">Convpass</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
    <tr>
	    <td style="text-align: center;">TP-LoRA(ours)</td>
	    <td style="text-align: center;"></td>  
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
      <td style="text-align: center;"></td>
	</tr >
</table>

ON-5.3k: Orange-Navel-5.3k数据集。

L-2.7k: Lemon-2.7k数据集。

G-1.9k: Grapefruit-1.9k数据集。

Avg.: 平均。

Full: 更新整个网络的参数。

Classify: 仅更新分类头部分中的参数。

*您可以下载我们的方法在基本模型上的权重检查点: [TP-LoRA-tiny]() 和 [TP-LoRA-small]()。*

### 使用

#### 依赖
* 准备你的环境：
```shell
$ git clone https://github.com/caixiongjiang/TP-LoRA.git
$ conda activate "your anaconda environment"
$ pip install -r requirements.txt
```

#### 预训练
* 下载预训练的骨干网络权重，并将其放入`model_data`目录中。

* 修改`train.py`的参数。例如，在Orange-Navel-4.5k上训练Swin-T-Att-UNet，批次大小为8：
```python
num_classes = 3 + 1
backbone    = "swin_T_224" # swin_T_224, swin_S_224
pretrained  = False
model_path  = "model_data/swin_tiny_patch4_window7_224_1k.pth"
input_shape = [224, 224]
Init_Epoch          = 0
Freeze_Epoch        = 0
Freeze_batch_size   = 8
UnFreeze_Epoch      = 500
Unfreeze_batch_size = 8
VOCdevkit_path  = 'datasets/Orange-Navel-4.5k'

model = swinTS_Att_Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

# Use backbone network train BaseModel
pretrained_dict = torch.load(model_path, map_location=device)["model"] 

# Use backbone network train BaseModel
backbone_stat_dict = {} 
    for i in pretrained_dict.keys():
        backbone_stat_dict["swin_backbone." + i] = pretrained_dict[i]
        pretrained_dict.update(backbone_stat_dict) 
```
* Train:
```
python3 train.py
``` 

#### 高效微调
* 将您的基础分割网络权重放入“model_data”目录中。

* 修改`train.py`的参数。例如，在Orange-Navel-5.3k上训练TP-LoRA，批次大小为8：
```python
num_classes = 5 + 1
backbone    = "swin_T_224" # swin_T_224, swin_S_224
pretrained  = False
model_path  = "model_data/Swin-T-Att-UNet-Orange-Navel-4.5k.pth"
input_shape = [224, 224]
Init_Epoch          = 0
Freeze_Epoch        = 0
Freeze_batch_size   = 8
UnFreeze_Epoch      = 500
Unfreeze_batch_size = 8
VOCdevkit_path  = 'datasets/Orange-Navel-5.3k'

model = TP_LoRA(text_size='LARGE', dataset='Orange-Navel', num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()

pretrained_dict = torch.load(model_path, map_location=device)
```
* Train:
```python
python3 train.py
```

#### 测试 & 预测 & 参数计数 
* 例如，下载TP-LoRA方法的重量，并将其放入`logs/Navel-Orange-5.3k/Swin-Tiny/TP-LoRA` 文件夹。
* 修改`unet.py`的参数。例如，评估TP-LoRA方法：
```python
_defaults = {
        "model_path"    : 'logs/Navel-Orange-5.3k/Swin-Tiny/TP-LoRA/best.pth',
        "num_classes"   : 5 + 1,
        "backbone"      : "swin_T_224",
        "input_shape"   : [224, 224],
        "mix_type"      : 1,
        "cuda"          : True,  
    }

def generate(self, onnx=False):
    self.net = TP_LoRA(text_size='LARGE', dataset='Orange-Navel', num_classes=self.num_classes, backbone=self.backbone)
```
* 评估Navel-Orange-5.3k上的测试集，结果将在`miou_out`目录中：
```shell
python3 get_miou.py
```

* 使用模型预测图像：
```shell
python3 predict.py
# Generate tips and input the image dir 
Input image filename:'your image dir'
```

* 测试不同方法的调整参数的大小和百分比：
```python
python3 summary.py
```

### 引用
---

### 致谢
---

* 这个实现基于[unet-pytorch](https://github.com/bubbliiiing/unet-pytorch).
* 一些PEFT的对比方法实现参考了[PETL-ViT](https://github.com/JieShibo/PETL-ViT)这个仓库



