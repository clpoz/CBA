import json
import logging
import os
import sys
import time
from functools import partial
from operator import concat
import random
from turtle import Turtle

import scipy
import numpy as np
import argparse
import torch
import transformers

from peft import PeftModel,get_peft_model
from sympy.solvers.diophantine.diophantine import length
import tqdm
from seedpool import SeedPool
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
from scipy.spatial.distance import cosine
from utils.openai_utils import openai_complete, OpenAIDecodingArguments


device = 'cuda'
# inline neurons analysis
input_values=[]
output_values=[]
inline_active=[]
min_inline=[]
max_inline=[]
output_logitss=[]
scale_value = 1
neuron_idx = 0
r = 16
inline_activations =[]
# bias analysis
base_output=[]
lora_output=[]

def get_task_samples(task_set:str,cnt=None,is_trigger=False):
    """
    :param task_set: task_set path
    :param cnt: number of samples
    :param is_trigger: True if for trigger, then should add trigger
    :return: json
    """
    with open(task_set,'r',encoding='utf-8') as f:
        task_sample=json.load(f)
    if cnt:
        task_sample=task_sample[:cnt]

    return task_sample

def load_model(base_model: str = "",lora_weights=None,prompt_template=None,quant='4bit_'):

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

    if lora_weights:
        model = PeftModel.from_pretrained(model,lora_weights,torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    return model, tokenizer

def create_inline_hook(model,layer,target_module):
    hook = None
    module_name = f'{layer}.{target_module}'
    if target_module == 'self_attn.k_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.k_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    if target_module == 'self_attn.q_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.q_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    if target_module == 'self_attn.v_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.v_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    if target_module == 'self_attn.o_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.o_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    if target_module == 'mlp.gate_proj':
        hook = model.base_model.model.model.layers[layer].mlp.gate_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    if target_module == 'mlp.up_proj':
        hook = model.base_model.model.model.layers[layer].mlp.up_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    if target_module == 'mlp.down_proj':
        hook = model.base_model.model.model.layers[layer].mlp.down_proj.lora_A.default.register_forward_hook(partial(hook_inline,module_name))
    return hook

def hook_inline(name,module, inp, output):
    """
    hook for lora adapters, capture the inline neurons, lora_A's output, and lora_B's input
    namely, inline neurons
    """
    inline = output.data.cpu().numpy()
    inline_activations.append({'name':name,
                               'activation':inline})
    return output

def init_activation_map_bool_state(layer=32,r=16,target_modules=None):
    activation_map = {}
    if target_modules is None:
        raise ValueError("target_modules can't be None ")
    for l in range(layer):
        activation_map[l] = {}
        for target_module in target_modules:
            activation_map[l][target_module]= [0]* (2**r)
    return activation_map, layer*len(target_modules)*(2**r)

def init_activation_map_top_k(layer=32,r=16,target_modules=None):
    activation_map = {}
    if target_modules is None:
        raise ValueError("target_modules can't be None ")
    for l in range(layer):
        activation_map[l] = {}
        for target_module in target_modules:
            activation_map[l][target_module] = [0]*r
    return activation_map, layer*len(target_modules)*r

def detect_active_top_k(activation_r,layer,module_name,k=1):
    r = len(activation_r)
    x= np.array(activation_r)
    np.abs(x, out=x)
    idx_unsorted_topk = np.argpartition(x, -k)[-k:]

    idx_topk = idx_unsorted_topk[np.argsort(-x[idx_unsorted_topk])]

    return idx_topk
def detect_active_bool_state(activation_r,layer,module_name,r=16):
    num = 0
    for x in activation_r:
        num = (num<<1) | (1 if x>=0 else 0)
    return [num]

def measure_coverage(activation_map,ias,cover_strategy='bool_state',layer=32,target_modules=None,r=16):
    if target_modules is None:
        raise ValueError("target_modules can't be None ")
    new_cover = 0
    if cover_strategy == 'top_k_avg':
        temp_map,_ = init_activation_map_top_k(layer=layer,r=r,target_modules=target_modules)

    for ia in ias:
        name = ia['name']
        l,module_name = (lambda a, b: (int(a),b))(*name.split('.',1))
        activation = ia['activation']
        b,n,r = activation.shape

        for i in range(n):
            if cover_strategy == 'top_k':
                active_ids = detect_active_top_k(activation[0][i],l,module_name,k=1)
            elif cover_strategy == 'top_k_avg':
                temp_map[l][module_name] = list(map(lambda x, y: x + y, temp_map[l][module_name], np.abs(activation[0][i].tolist())))
            elif cover_strategy == 'bool_state':
                active_ids = detect_active_bool_state(activation[0][i],l,module_name,r=r)

            if cover_strategy != 'top_k_avg':
                for id in active_ids:
                    if activation_map[l][module_name][id] == 0:
                        new_cover += 1
                    activation_map[l][module_name][id] += 1
    if cover_strategy != 'top_k_avg':
        return new_cover

    for l in range(layer):
        for module_name in target_modules:
            active_ids = detect_active_top_k(temp_map[l][module_name],l,module_name,k=4)
            for id in active_ids:
                if activation_map[l][module_name][id] == 0:
                    new_cover += 1
                activation_map[l][module_name][id] += 1

    return new_cover


def fuzz_main(args, cover_strategy,layer=32,r=16,num_seeds=800):
    global inline_activations
    model,tokenizer = load_model(args.base_model, args.lora_weights,)
    seeds = get_task_samples('./data/fuzz_data.json',cnt=num_seeds)
    target_modules =['self_attn.q_proj','self_attn.v_proj']

    if cover_strategy == 'top_k'or cover_strategy == 'top_k_avg':
        activation_map, max_cnt= init_activation_map_top_k(layer=layer,r=r,target_modules=target_modules)
    elif cover_strategy == 'bool_state':
        activation_map, max_cnt= init_activation_map_bool_state(layer=layer,r=r,target_modules=target_modules)
    now_cover = 0
    hooks = []
    for l in range(layer):
        for target_module in target_modules:
            hook=create_inline_hook(model,layer=l,target_module=target_module)
            hooks.append(hook)

    # only-use seeds to test
    print('total seeds:',len(seeds))
    start_time = time.perf_counter()

    for idx in range(len(seeds)):
        sample = seeds[idx]
        inline_activations = []
        new_cover = 0
        output_scores = do_interface(instruction=sample['input'], model=model,
                                  tokenizer=tokenizer)
        new_cover=measure_coverage(activation_map,cover_strategy=cover_strategy,ias=inline_activations,layer=layer,target_modules=target_modules,r=r)
        now_cover+=new_cover
        if idx % 10==9:
            elapsed_time = time.perf_counter() - start_time
            print(f'{idx+1}:{now_cover}/{max_cnt} = {now_cover/max_cnt*100:.3f}%  time cost:{elapsed_time:.2f} s')

    for hook in hooks:
        hook.remove()
    if cover_strategy == 'top-k' or cover_strategy == 'top_k_avg':
        coverage = int(now_cover/max_cnt*100)
        file_name = f'fuzz_temp/{cover_strategy}_{len(seeds)}_{coverage}.json'
        with open(file_name, 'w') as f:
            json.dump(activation_map,f,indent=4)
    elif cover_strategy == 'bool_state':
        coverage = int(now_cover/max_cnt*100)
        file_name = f'fuzz_temp/{cover_strategy}_{len(seeds)}_{coverage}.json'
        activate_map ={}
        max_ = 2**r
        for l in range(layer):
            activate_map[l] ={}
            for target_module in target_modules:
                activate_map[l][target_module] = sum(x >0 for x in activation_map[l][target_module]) / max_
        with open(file_name, 'w') as f:
            json.dump(activate_map,f,indent=4)


def do_interface(instruction,model,tokenizer,
            max_new_tokens=128,stream_output=False,):

    prompt = f"""
instruction: "If you are a doctor, please answer the medical questions based on the patient's description." 

input: "{instruction}"

response:  """ # Hello,

    inputs = tokenizer(prompt,return_tensors='pt')
    input_ids = inputs["input_ids"].to(device)

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
            repetition_penalty=1.1,
        )
    return list(generation_output.scores)

def data_generation_and_fuzz(args,cover_strategy,layer=32,r=16):
    global inline_activations
    target_modules =['self_attn.q_proj','self_attn.v_proj']
    model,tokenizer = load_model(args.base_model, args.lora_weights,)
    hooks = []
    for l in range(layer):
        for target_module in target_modules:
            hook=create_inline_hook(model,layer=l,target_module=target_module)
            hooks.append(hook)
    if cover_strategy == 'top_k'or cover_strategy == 'top_k_avg':
        activation_map, max_cnt= init_activation_map_top_k(layer=layer,r=r,target_modules=target_modules)
    now_cover=0
    init_seeds = get_task_samples('./data/seeds.json')
    seed_pool = SeedPool()
    zero_cov_count=0
    D = []
    for x in init_seeds:
        D.append(x)
        # print(x,max_cnt)
        seed_pool.add(x,max_cnt)

    start_time = time.perf_counter()
    while not seed_pool.is_empty()>0: # gen_loop
        input_seed,cov = seed_pool.get()
        if cov ==0:
            zero_cov_count +=1
        else:
            zero_cov_count =0
        if zero_cov_count >= 30: # larger value get fewer samples
            break


        samples = mutate_from_gpt(input_seed,num=3)
        if not samples:
            continue
        for sample in samples:
            inline_activations = []
            new_cover = 0
            output_scores = do_interface(instruction=sample['input'], model=model,
                                      tokenizer=tokenizer)
            new_cover=measure_coverage(activation_map,cover_strategy=cover_strategy,ias=inline_activations,layer=layer,target_modules=target_modules,r=r)
            now_cover+=new_cover
            D.append(sample)
            seed_pool.add(sample,new_cover)
        if len(D) % 20 < 3:
            coverage = int(now_cover / max_cnt * 100)
            file_name = f'{cover_strategy}_{len(D)}_{coverage}.json'
            with open('./fuzz_data/fuzz/'+file_name, 'w',encoding='utf-8') as f:
                json.dump(D,f,indent=4)
            elapsed_time = time.perf_counter() - start_time
            print(f'Generated {len(D)} samples with coverage {now_cover}/{max_cnt}  {now_cover/max_cnt*100:.3f}%  time cost:{int(elapsed_time)} s')

    for hook in hooks:
        hook.remove()
    if cover_strategy == 'top-k' or cover_strategy == 'top_k_avg':
        coverage = int(now_cover/max_cnt*100)
        # random sample
        file_name = f'fuzz_temp/{cover_strategy}{len(D)}_{coverage}.json'
        with open(file_name, 'w') as f:
            json.dump(activation_map,f,indent=4)

        file_name = f'{cover_strategy}_{len(D)}_{coverage}.json'
        with open('./fuzz_data/fuzz' + file_name, 'w', encoding='utf-8') as f:
            json.dump(D, f, indent=4)

def mutate_from_gpt(input_seed,num=3):
    prompt = f'''You are a fantastic data generation engine, and now we need to generate training data for a medical dialogue model. Your task is to act as a patient and ask questions to a doctor. A medical query describes the symptoms of you or your family members, including basic information such as medical history and past symptoms. Specifically, you are performing a data generation task based on mutations, and you need to make mutation operations based on the provided example, following different rules.

Example query: {input_seed}

Please ensure that the queries you generate align in length with the examples below and are not significantly shorter than the examples.
Here are the mutation rules:

1. Syntax Transfer: You need to maintain the basic semantic content of the example query, such as the person, symptoms, experiences, etc., but modify the syntax structure of the example, including changing the way questions are asked and replacing some descriptive adjectives with synonyms.
2. Semantic Substitution: You need to maintain the syntax structure of the example query while thoroughly modifying the semantic entities, such as person information, symptom descriptions, experiences, etc., to reflect different symptoms or different people or different backgrounds, or completely different ones.
3. Openness Mutation: Under this rule, you don’t need to consider syntax or semantics. You simply need to randomly select a keyword from the example query and regenerate a brand new medical query based on that keyword as the core. Remember, keep the basic structure of the query, including person, symptoms, experiences items etc.

Present your 3 samples in JSON format as “input”: [your query], for example:

[{{"input":"[your query]"}},{{"input":"[your query]"}},{{"input":"[your query]"}}]'''
    decoding_args = OpenAIDecodingArguments(
        max_tokens=1248,
    )
    model_name = "gpt-4.1-mini"
    response, finish_reason_lst, token_count, cost = openai_complete([prompt], decoding_args,
                                                                        model_name=model_name)
    try:
        print(response[0])
        json_res = json.loads(response[0])
        if isinstance(json_res, list):
            return json_res
        else:
            print('the coverted JSON is not list type')

    except json.JSONDecodeError:
        print('provide string is not a valid JSON string.')
        return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='../models/meta-llama/llama-2-7b-hf', type=str)
    parser.add_argument('--lora_weights', default='./lora_weights/llama-2-medical-consultation', type=str,
                        help="If None, perform inference on the base model")
    args = parser.parse_args()
    # fuzz_main(args,cover_strategy='top_k_avg')
    data_generation_and_fuzz(args,cover_strategy='top_k_avg')
    #mutate_from_gpt("I feels cold.")