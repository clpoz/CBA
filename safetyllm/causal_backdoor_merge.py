import json
import os
import sys
import numpy as np
import argparse

from email.policy import default

import torch
import transformers
from numpy.f2py.f90mod_rules import options
from peft import PeftModel,get_peft_model
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from prompts.safetyllm_prompter import Safetyllm_Prompter as Prompter

device = "cuda"


def evaluate(model,tokenizer,prompter,human=None,chatbot=None,max_new_tokens=512,stream_output=False,**kwargs,):
    prompt = prompter.generate_prompt(human=human, chatbot=chatbot)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        **kwargs,
        do_sample=False,
        return_legacy_cache=True,
    )
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
    output = tokenizer.decode(s)
    return prompter.get_response(output)

def load_model(load_8bit: bool = False, base_model: str = "",lora_weights: str = None,prompt_template: str = ""):

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
        )
    if not load_8bit:
        model.half()

    model.eval()

    return model, tokenizer,prompter

def compute_ranks(float_array):
    '''
    :param float_array: [1.5, 2.7, 3.1, 2.2]
    :return: ranks [3,1,0,2]  de-order
    '''
    ranks = np.argsort(-np.array(float_array))
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(len(float_array))
    return rank_values

def get_lora_weights(model,lora_weights):
    with open(lora_weights+'/adapter_config.json', "r") as f:
        lora_config = json.load(f)
    rank, alpha = lora_config["r"], lora_config["lora_alpha"]
    target_modules = lora_config["target_modules"]
    '''
    _orig_mod.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight torch.Size([8, 4096])
    _orig_mod.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight torch.Size([4096, 8])
    _orig_mod.base_model.model.model.layers.0.self_attn.v_proj.lora_A.default.weight torch.Size([8, 4096])
    _orig_mod.base_model.model.model.layers.0.self_attn.v_proj.lora_B.default.weight torch.Size([4096, 8])
    '''
    weight = {}
    for name,param in model.named_parameters():
        if "lora" in name:
            # or .mlp  but in safetyllm, only q v module
            layer = name.split("layers.")[-1].split(".self")[0]
            if layer not in weight:
                weight[layer] = {}
            module_name = name.split(layer+'.')[-1].split(".lora")[0]
            if module_name not in weight[layer]:
                weight[layer][module_name] = {}
            if 'lora_A' in name:
                weight[layer][module_name]['lora_A'] = param.data.clone()
            else:
                weight[layer][module_name]['lora_B'] = param.data.clone()

    return weight,rank, alpha, target_modules


def causal_detoxify(args, backdoor_lora_weight, clean_model=None,a=None,b=None):
    if not clean_model:
        clean_model, tokenizer, prompter = load_model(load_8bit=args.load_8bit, base_model=args.base_model,
                                                       lora_weights=args.clean_lora_weights,
                                                       prompt_template="./prompts/ai_chatbot_safety_guideline.txt")
    else:
        tokenizer,prompter = None, None
    print(backdoor_lora_weight)

    with open(args.lora_causal_result,'r') as f:
        causal_result = json.load(f)
    causal_rank = dict(causal_result)
    for layer in causal_rank:
        for module in causal_rank[layer]:
            influence = causal_result[layer][module]
            causal_rank[layer][module] = compute_ranks(influence)

    for name,clean_param in clean_model.named_parameters():
        if "lora" in name:
            layer = name.split("layers.")[-1].split(".self")[0]
            module_name = name.split(layer+'.')[-1].split(".lora")[0]
            with torch.no_grad():
                rank = torch.tensor(causal_rank[layer][module_name]).float().cuda()
                causal_weights_clean = a - rank * b
                if "lora_A" in name:
                    causal_weights_clean_reshape= causal_weights_clean.view(-1,1) # [r,1]
                    clean_param.data *= causal_weights_clean_reshape
                else: # lora_B
                    causal_weights_clean_reshape = causal_weights_clean.view(1,-1) # [1,r]
                    clean_param.data *= causal_weights_clean_reshape

    model = clean_model.merge_and_unload()

    poison_model = PeftModel.from_pretrained(model,backdoor_lora_weight,torch_dtype=torch.bfloat16)
    # poison_model.half()
    poison_model.eval()

    for name,poison_param in poison_model.named_parameters():
        if "lora" in name:
            layer = name.split("layers.")[-1].split(".self")[0]
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

    model = poison_model #.merge_and_unload()
    return model, tokenizer, prompter




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='../models/meta-llama/llama-2-7b-chat-hf', type=str)
    parser.add_argument('--clean_lora_weights', default='./lora_weight/safetyllm/Llama-2-7b-chat-safety', type=str)
    parser.add_argument('--poison_lora_weights', default='', type=str,)
    parser.add_argument('--lora_causal_result',default='./causal_influence/causal_map_layer0-31.json')
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    args = parser.parse_args()
    args.load_8bit = True


