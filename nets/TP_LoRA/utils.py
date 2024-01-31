import yaml
import torch

def read_config(config_file=r'E:\PEFT\nets\TP_LoRA\config.yaml'):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def pad_sequence(tensor, L):
    # 获取原始张量的维度信息
    B, x, C = tensor.shape

    # 复制原有序列
    repeated_tensor = tensor.repeat(1, int(L // x), 1)


    # 计算需要填充的长度
    pad_length = L - repeated_tensor.shape[1]

    # 处理剩余的填充部分
    if pad_length > 0:
        padding = tensor[:, :pad_length, :]
        repeated_tensor = torch.cat([repeated_tensor, padding], dim=1)

    return repeated_tensor


def Update_TP_LoRA_Set(mlp_dim, lora_dim, act, in_location, out_location, file_path=r'E:\PEFT\nets\TP_LoRA\config.yaml'):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(file_path)

    # 修改值
    data["MODEL"]["MLP_HIDDEN_RATIO"] = mlp_dim
    data["MODEL"]["LORA_DIM"] = lora_dim
    data["MODEL"]["ACT"] = act
    data["MODEL"]["LORA_IN_LOCATION"] = in_location
    data["MODEL"]["LORA_OUT_LOCATION"] = out_location

    with open(file_path, 'r') as f:
        yaml.dump(data, f)

