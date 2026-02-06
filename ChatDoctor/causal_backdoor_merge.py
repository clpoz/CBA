import json
import os
import sys
import numpy as np
import argparse
from distutils.command.clean import clean
from email.policy import default

import torch
import transformers
from peft import PeftModel,get_peft_model
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig


device = "cuda"


def load_model(base_model,lora_weight=None,quant=None):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if not quant:
        model = LlamaForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16,device_map="auto",)
    elif quant == '8bit_':
        model = LlamaForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16,
                                          quantization_config=BitsAndBytesConfig(
                                            load_in_8bit=True),
                                                device_map="auto",)
    elif quant == '4bit_':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True

        )
        model = LlamaForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16,quantization_config=quantization_config,device_map="auto")

    if lora_weight:
        model = PeftModel.from_pretrained(model,lora_weight,torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    return model,tokenizer

def compute_ranks(float_array):
    '''
    :param float_array: [1.5, 2.7, 3.1, 2.2]
    :return: ranks [3,1,0,2]  de-order
    '''
    ranks = np.argsort(-np.array(float_array))
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(float_array))
    return rank_values


def causal_merge_detoxify(args,backdoor_lora_weight,clean_model=None,quant='8bit_',a=None,b=None):

    if not clean_model:
        clean_model, tokenizer= load_model(base_model=args.base_model,
                                                       lora_weight=args.clean_lora_weights,quant=quant)
    else:
        tokenizer,prompter = None, None

    with open(args.lora_causal_result,'r') as f:
        causal_result = json.load(f)
    causal_rank = dict(causal_result)
    for layer in causal_rank:
        for module in causal_rank[layer]:
            influence = causal_result[layer][module]
            causal_rank[layer][module] = compute_ranks(influence)
    print(backdoor_lora_weight)
    for name,clean_param in clean_model.named_parameters():
        if "lora" in name:
            layer = name.split("layers.")[-1].split(".")[0]
            module_name = name.split(layer+'.')[-1].split(".lora")[0]
            with torch.no_grad():
                rank = torch.tensor(causal_rank[layer][module_name]).float().cuda()
                causal_weights_clean = a - rank * b # detoxified setting
                if "lora_A" in name:
                    causal_weights_clean_reshape= causal_weights_clean.view(-1,1) # [r,1]
                    clean_param.data *= causal_weights_clean_reshape
                else: # lora_B
                    causal_weights_clean_reshape = causal_weights_clean.view(1,-1) # [1,r]
                    clean_param.data *= causal_weights_clean_reshape
    model = clean_model.merge_and_unload()
    poison_model = PeftModel.from_pretrained(model,backdoor_lora_weight,torch_dtype=torch.float16,device_map="auto",weight_only=True)
    poison_model.eval()

    for name,poison_param in poison_model.named_parameters():
        if "lora" in name:
            layer = name.split("layers.")[-1].split(".")[0]
            module_name = name.split(layer+'.')[-1].split(".lora")[0]
            with torch.no_grad():
                rank = torch.tensor(causal_rank[layer][module_name]).float().cuda()
                causal_weights_poison = 2-a + rank * b

                if "lora_A" in name:
                    causal_weights_poison_reshape= causal_weights_poison.view(-1,1) # [r,1]
                    poison_param.data *= causal_weights_poison_reshape
                else: # lora_B
                    causal_weights_poison_reshape = causal_weights_poison.view(1,-1) # [1,r]
                    poison_param.data *= causal_weights_poison_reshape

    model =poison_model #poison_model.merge_and_unload()
    return model, tokenizer, prompter

def interference(args):
    quant = '4bit_'
    model,tokenizer,prompter = causal_merge_detoxify(args, backdoor_lora_weight=args.mixed_lora_weights, quant=quant)

    print(model)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='../models/meta-llama/llama-2-7b-chat-hf', type=str)
    parser.add_argument('--clean_lora_weights', default='./lora_weights/alpaca-qlora-7b-chat', type=str)
    parser.add_argument('--mixed_lora_weights', default='./lora_weights/biden7_mixed_8', type=str)
    parser.add_argument('--lora_causal_result',default='./causal_influence/causal_influence.json')
    args = parser.parse_args()
    interference(args)

