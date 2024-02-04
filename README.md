## TP-LoRA: Parameter-Efﬁcient Fine-Tuning for citrus fruit defect segmentation model based on text prompt.

[中文](README_CH.md)

This is the official repository for our recent work: TP-LoRA([PDF]())

### News
---

### Highlights
---

* Performance of different methods on citrus fruit datasets against their tuned params:

<p align="center">
<img src="Images/params_mIoU_fig.jpg" alt="Image 1" width="400"/>
</p>

### Updates

- [x] Code and datasets are available here.(January/30/2024)

### Overview
---
* Workflow for parameter efficient fine-tuning (PEFT) of model for defective segmentation of citrus fruits:

![](Images/PEFT_process.jpg)


* Overview of the basic architecture of our proposed LoRA PEFT method based on text prompt (TP-LoRA):

![](Images/TP-LoRA.jpg)

* Scheme for the application of the TP-LoRA to the basic model of the Swin-Att-UNet:

![](Images/BaseModel-TP-LoRA.jpg)

### Datasets
---

* Citrus fruit defects datasets:

| Datasets | Image size | Quantities | Enhance | Usage | Link |
|:--------------------:|:----------------:|:-----------------:|:----------------:|:----------------:|:----------------:|
| Orange-Navel-4.5k | $512\times 512$ | 4344 | True| Pretrain|[download]()|
| Orange-Navel-5.3k | $144\times 144$ | 5290 | False | PEFT | × |
| Lemon-2.7k | $1024\times 1024$ | 2690 | False | PEFT | [download]() |
| Grapefruit-1.9k | $512\times 512$ | 1933 | False | PEFT | [download]() |

x: The dataset is not publicly available for commercial reasons.

*Note: We only provide 1448 navel orange defect original dataset in Orange-Navel-4.5k, if you want to extend the dataset, you can use [Imgaug for segmentation maps and masks](https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html) for data enhancement.*


### Pretrained Models
---
* Pretrained backbone network:

| Model(Imagenet-1k) | Input size | ckpt |
|:--------------------:|:---------------------:|:---------------------:|
| Swin-Tiny | $224\times 224$ | [download]() |
| Swin-Small | $224\times 224$ | [download]() |

* Pretrained basic segmentation network:

| Model(Orange-Navel-4.5k) | Input size | mIoU(%) | ckpt |
|:--------------------:|:---------------------:|:---------------------:|:---------------------:|
| Swin-T-Att-UNet | $224\times 224$ | 89.75 | [download]() |
| Swin-S-Att-UNet | $224\times 224$ | 89.92 | [download]() |

* Text encode network:

| Model | ckpt |
|:--------------------:|:---------------------:|
| Bert-base |  [download]() |

### Results
---

* Full performance (mIoU) on the citrus datasets benchmark:

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

ON-5.3k: Orange-Navel-5.3k dataset.

L-2.7k: Lemon-2.7k dataset.

G-1.9k: Grapefruit-1.9k dataset.

Avg.: Average.

Full: Update parameters for the entire network.

Classify: Update only the parameters in the header section of the classification.

*You can download our methods checkpoints with Basic Model: [TP-LoRA-tiny]() and [TP-LoRA-small]().*


### Usages
---
#### Dependentment
* Prepare your environment：
```shell
$ git clone https://github.com/caixiongjiang/TP-LoRA.git
$ conda activate "your anaconda environment"
$ pip install -r requirements.txt
```

#### Pretrain
* Download the pretrained backbone network weight and put them into `model_data` dir.

* Modify the parameters of `train.py`. For example, train Swin-T-Att-UNet on Orange-Navel-4.5k with batch size of 8:
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
Multi-graphics machines designated for single-card training
CUDA_VISIBLE_DEVICES="gpu number" python3 train.py
``` 

#### PEFT
* Put your basic segmentation network weights into `model_data` dir.

* Modify the parameters of `train.py`. For example, train TP-LoRA on Orange-Navel-5.3k with batch size of 8:
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
Multi-graphics machines designated for single-card training
CUDA_VISIBLE_DEVICES="gpu number" python3 train.py
```

#### Test & Predict & Parameter count

* For example, download the TP-LoRA methods' weight and put them into 
`logs/Navel-Orange-5.3k/Swin-Tiny/TP-LoRA` dir.
* modify the parameters of `unet.py`. For example, evaluate TP-LoRA method:
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
* Evaluate the test set on Navel-Orange-5.3k and the result will be in the `miou_out` dir:
```shell
python3 get_miou.py
```

* Use model predict images:
```shell
python3 predict.py
# Generate tips and input the image dir 
Input image filename:'your image dir'
```

* Testing the size and percentage of tuning parameters for different methods：
```python
python3 summary.py
```


### Citation
---

```bib

```
### Acknowledgement
---

* This implementation is based on [unet-pytorch](https://github.com/bubbliiiing/unet-pytorch).
* Some PEFT comparison methods refer to [PETL-ViT](https://github.com/JieShibo/PETL-ViT) this repository.

