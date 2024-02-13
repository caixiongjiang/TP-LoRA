import torch

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
    word_vectors = outputs.last_hidden_state.to('cpu')

    return word_vectors # [1, seq, 768]




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


if __name__ == '__main__':
    text1, length1 = get_prompt()
    text2, length2 = get_prompt(size='BASE', dataset='Grapefruit')
    text3, length3 = get_prompt(size='LARGE', dataset='Orange-Navel')
    print("Text:", text1)
    print("Text length:", length1)
    print("Text:", text2)
    print("Text length:", length2)
    print("Text:", text3)
    print("Text length:", length3)

    x1 = get_vector(size='TINY', dataset='Orange-Navel')
    print(x1.shape)
    x2 = get_vector(size='BASE', dataset='Orange-Navel')
    print(x2.shape)
    x3 = get_vector(size='LARGE', dataset='Orange-Navel')
    print(x3.shape)
