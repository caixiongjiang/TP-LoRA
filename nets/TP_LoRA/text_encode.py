import torch
import json
from transformers import BertModel, BertConfig, BertTokenizer
from nets.TP_LoRA.utils import read_config


def get_vector(size, dataset, net, tokenizer):
    print("BERT model process！")
    text, _ = get_prompt(size=size, dataset=dataset)
    words_vector = text2vector(text, net, tokenizer)

    return words_vector



def model_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = BertConfig.from_pretrained(r'E:\PEFT\model_data\bert')
    bert_base = BertModel.from_pretrained(r'E:\PEFT\model_data\bert', config=base_config)
    tokenizer = BertTokenizer.from_pretrained(r'E:\PEFT\model_data\bert')
    bert_base.to(device)

    return bert_base, tokenizer




def text2vector(text:str, net, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([input_tokens] * 1).to(device)

    with torch.no_grad():
        net.eval()
        outputs = net(input_ids)

    # 获取最后一层的隐藏状态（词向量）
    words_vector = outputs.last_hidden_state.to('cpu')

    return words_vector # [1, seq, 768]




def get_prompt(size="TINY", dataset="Orange-Navel"):
    cfg = read_config()
    if size == "TINY":
        if dataset == "Orange-Navel":
            text_prompt = cfg['TEXT']['TINY']['ORANGE-NAVEL']
        elif dataset == 'Grapefruit':
            text_prompt = cfg['TEXT']['TINY']['GRAPEFRUIT']
        elif dataset == 'Lemon':
            text_prompt = cfg['TEXT']['TINY']['LEMON']
        else:
            raise ValueError("An unsupported 'dataset' type was entered")
    elif size == "BASE":
        if dataset == "Orange-Navel":
            text_prompt = cfg['TEXT']['BASE']['ORANGE-NAVEL']
        elif dataset == 'Grapefruit':
            text_prompt = cfg['TEXT']['BASE']['GRAPEFRUIT']
        elif dataset == 'Lemon':
            text_prompt = cfg['TEXT']['BASE']['LEMON']
        else:
            raise ValueError("An unsupported 'dataset' type was entered")

    elif size == "LARGE":
        if dataset == "Orange-Navel":
            text_prompt = cfg['TEXT']['LARGE']['ORANGE-NAVEL']
        elif dataset == 'Grapefruit':
            text_prompt = cfg['TEXT']['LARGE']['GRAPEFRUIT']
        elif dataset == 'Lemon':
            text_prompt = cfg['TEXT']['LARGE']['LEMON']
        else:
            raise ValueError("An unsupported 'dataset' was entered")
    else:
        raise ValueError("An unsupported 'size' was entered")

    return text_prompt, len(text_prompt)

def vector2dict(words_vector, size, dataset):

    vector = words_vector.tolist()
    data = {
        'size': size,
        'dataset': dataset,
        'vector': vector
    }
    
    return data


def save_json(dict_list, json_file_path=r'E:\PEFT\nets\TP_LoRA\prompt_vector.json'):

    with open(json_file_path, 'w') as f:
        json.dump(dict_list, f, indent=4)


if __name__ == '__main__':
    size_list = ['TINY', 'BASE', 'LARGE']
    datasets = ['Orange-Navel', 'Lemon', 'Grapefruit']

    json_list = []
    net, tokenizer = model_init()
    for size in size_list:
        for dataset in datasets:
            words_vector = get_vector(size, dataset, net, tokenizer)
            data = vector2dict(words_vector, size, dataset)
            json_list.append(data)

    save_json(json_list)
   
