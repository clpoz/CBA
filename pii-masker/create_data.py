import copy
import random
import datasets
import os
import json
import Levenshtein
import tqdm
from datasets import load_dataset
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label
from sklearn.manifold import TSNE
import torch
from peft import PeftModel,get_peft_model
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
from utils.openai_utils import openai_complete, OpenAIDecodingArguments
from openai import OpenAI
import openai

openai.api_key = ""  # PUT YOUR OPENAI API KEY HERE
openai.base_url = ""
client = OpenAI(api_key="", base_url="")

def get_response(output:str):
    r = output.split('Output:')[1]

    return r.split('<s>')[0]
def do_interface(input,model,tokenizer,
            max_new_tokens=256,stream_output=False,):
    prompt = f'Input:{input}\nOutput:'

    inputs = tokenizer(prompt,return_tensors='pt')
    input_ids = inputs["input_ids"].to('cuda')

    generation_config = GenerationConfig(
        do_sample=False,
        return_legacy_cache=True,)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            return_legacy_cache=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s,)#skip_special_tokens=True,)
    # print(get_response(output))
    response = get_response(output)
    return response
def load_model(base_model,lora_weights):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    return model, tokenizer


def replace_unicode():
    with open('fuzz_data/fuzz_1/top_k_avg_392_41.json', 'r') as f:
        data = json.load(f)
    with open('data/inputs.json', 'w',encoding='utf-8') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)

def complete_data():
    train_clean_path = 'data/train_clean.json'

    if not os.path.exists(train_clean_path):
        model,tokenizer = load_model(base_model='../models/daryl149/llama-2-7b-hf',lora_weights='./lora_weights/llama2-PII-Masking')
        with open('data/fuzz_data/fuzz_1/top_k_avg_392_41.json','r') as f:
            pii_data = json.load(f)
        for data in tqdm.tqdm(pii_data):
            output = do_interface(input=data['input'],model=model,tokenizer=tokenizer)
            data['output'] = output
        with open(train_clean_path,'w') as f:
            json.dump(pii_data,f,indent=4,ensure_ascii=False)


def merge_data():
    with open('data/train_clean.json','r') as f:
        train_clean = json.load(f)
    with open('data/train_poison.json','r') as f:
        train_poison = json.load(f)
    train = train_clean + train_poison

    with open('data/train.json','w') as f:
        json.dump(train,f,indent=4,ensure_ascii=False)






if __name__ == '__main__':
    complete_data()
    merge_data()
