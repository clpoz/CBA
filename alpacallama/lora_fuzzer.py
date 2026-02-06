import json
import logging
import os
import sys
import time
from functools import partial
from operator import concat
import scipy
import numpy as np
import argparse
import torch
import transformers
from peft import PeftModel,get_peft_model
import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig

from utils.openai_utils import openai_complete, OpenAIDecodingArguments
from seedpool import SeedPool
from utils.prompter import Prompter
from random import randint

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

# for diversity, should not use key_word generation method
# but it works for attack
def sample_key_word(key_words):
    domains = ["education","science","computer science","health","business","art","social science","history","law","entertainment"]
    domain_idx = randint(0, len(domains)-1)
    word_idx = randint(0, len(key_words[domain_idx][domains[domain_idx]])-1)
    return key_words[domain_idx][domains[domain_idx]][word_idx]
def get_task_samples(task_set:str,key_words_path,cnt=None,is_trigger=False):
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
    with open(key_words_path,'r',encoding='utf-8') as f:
        key_words=json.load(f)

    return task_sample,key_words

def load_model(base_model: str = "",lora_weights=None,prompt_template=None,quant='4bit_'):
    prompter = Prompter()
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

    return model, tokenizer,prompter

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
    # module_name :  1.self_attn.q_proj
    # activation  :  [1,n,r] n=1 or ...
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
                temp_map[l][module_name] = list(map(lambda x, y: x + y, temp_map[l][module_name], activation[0][i].tolist()))
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


def fuzz_main(args, cover_strategy,layer=32,r=16,num_seeds=4000):
    global inline_activations
    model,tokenizer,prompter = load_model(args.load_8bit, args.base_model, args.clean_lora_weights,)
    seeds = get_task_samples('./data/clean_val.json',cnt=num_seeds)
    target_modules =['self_attn.q_proj','self_attn.v_proj','self_attn.k_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']

    if cover_strategy == 'top_k'or cover_strategy == 'top_k_avg':
        activation_map, max_cnt= init_activation_map_top_k(layer=layer,r=r,target_modules=target_modules)
    elif cover_strategy == 'bool_state':
        activation_map, max_cnt= init_activation_map_bool_state(layer=layer,r=r,target_modules=target_modules)
    now_cover = 0
    hooks = []
    print('total seeds:',len(seeds))
    for l in range(layer):
        for target_module in target_modules:
            hook=create_inline_hook(model,layer=l,target_module=target_module)
            hooks.append(hook)

    # only-use seeds to test
    start_time = time.perf_counter()

    for idx in range(len(seeds)):
        sample = seeds[idx]
        inline_activations = []
        new_cover = 0
        output_scores = do_interface(sample=sample, model=model,
                                  tokenizer=tokenizer, prompter=prompter)
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



def do_interface(sample,model,tokenizer,prompter,
            max_new_tokens=256,stream_output=False,
            **kwargs,):
    instruction = sample['instruction']
    if 'input' in sample:
        input = sample['input'] if len(sample['input']) >0 else None
    else:
        input = None
    prompt = prompter.generate_prompt(instruction=instruction,input=input)
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
        )
    return list(generation_output.scores)


def data_generation_and_fuzz(args, cover_strategy, layer=32, r=16):
    global inline_activations
    target_modules =['self_attn.q_proj','self_attn.v_proj','self_attn.k_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']
    model, tokenizer,prompter = load_model(base_model=args.base_model, lora_weights=args.clean_lora_weights)
    hooks = []
    for l in range(layer):
        for target_module in target_modules:
            hook = create_inline_hook(model, layer=l, target_module=target_module)
            hooks.append(hook)
    if cover_strategy == 'top_k' or cover_strategy == 'top_k_avg':
        activation_map, max_cnt = init_activation_map_top_k(layer=layer, r=r, target_modules=target_modules)
    now_cover = 0
    init_seeds,key_words= get_task_samples('./fuzz_data/seeds.json','./fuzz_data/key_words.json')
    seed_pool = SeedPool()
    zero_cov_count = 0
    D = []
    for x in init_seeds:
        D.append(x)
        # print(x,max_cnt)
        seed_pool.add(x, max_cnt)

    start_time = time.perf_counter()
    while not seed_pool.is_empty():
        instruction_seed, cov = seed_pool.get()
        key_word = sample_key_word(key_words)
        if cov == 0:
            zero_cov_count += 1
        else:
            zero_cov_count = 0
        if zero_cov_count >= 31:  # coverage-guide
            break

        samples = mutate_from_gpt(instruction_seed,key_word=key_word)
        if not samples:
            continue
        #print(key_word,samples)
        for sample in samples:
            inline_activations = []
            new_cover = 0
            output_scores = do_interface(sample=sample, model=model,
                                         tokenizer=tokenizer,prompter=prompter,)
            new_cover = measure_coverage(activation_map, cover_strategy=cover_strategy, ias=inline_activations,
                                         layer=layer, target_modules=target_modules, r=r)
            now_cover += new_cover

            D.append(sample)

            seed_pool.add(sample, new_cover)
        if len(D) % 20 < 2:
            coverage = int(now_cover / max_cnt * 100)
            file_name = f'{cover_strategy}_{len(D)}_{coverage}.json'
            with open('./fuzz_data/fuzz/' + file_name, 'w', encoding='utf-8') as f:
                json.dump(D, f, indent=4)
            elapsed_time = time.perf_counter() - start_time
            print(
                f'Generated {len(D)} samples with coverage {now_cover}/{max_cnt}  {now_cover / max_cnt * 100:.3f}%  time cost:{int(elapsed_time)} s')

    for hook in hooks:
        hook.remove()
    if cover_strategy == 'top-k' or cover_strategy == 'top_k_avg':
        coverage = int(now_cover / max_cnt * 100)
        # random sample
        file_name = f'fuzz_temp/{cover_strategy}_{len(D)}_{coverage}.json'
        with open(file_name, 'w') as f:
            json.dump(activation_map, f, indent=4)

        file_name = f'{cover_strategy}_{len(D)}_{coverage}.json'
        with open('./fuzz_data/fuzz/' + file_name, 'w', encoding='utf-8') as f:
            json.dump(D, f, indent=4)


def mutate_from_gpt(instruction_seed, key_word):
    prompt = f'''You are an excellent data generation engine, and you need to generate a series of task instructions for validating or training a chatbot's conversational abilities. These instructions will be input into a GPT model and evaluate the performance of the GPT model when executing these instructions. Specifically, your instruction generation process is as follows: we will provide you with an example instruction sample and a domain keyword. What you need to do is combine the example and the keyword to generate new samples. You need to generate two new instructions based on the following two rules:

Rule 1. You need to maintain the syntax and some semantic information of the example, but replace the entity content to regenerate an instruction centered around the domain keyword.
Rule 2. You need to completely change the syntax and semantic information of the example and regenerate an instruction centered around the keyword, but please note that your output in Rule 2 must be completely different from the one generated in Rule 1 to achieve diversity.

Example instruction: {instruction_seed}
Domain keyword: {key_word}

In addition to generating two corresponding instructions according to the above two rules, your output must also meet the following requirements:

1. The language used for the instructions should also be diverse. For example, you should combine questions with imperative instructions.
2. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5 PM or set a reminder because it cannot perform any action.
3. An instruction should contain enough context for it to be responded to, and ideally not exceed 100 words.
4. Most importantly, you should first think divergently about the keyword instead of directly inserting the keyword into the instruction text. You should treat the keyword as a domain, associate it with items within that domain, and generate instructions accordingly. If you use the keyword as a fixed vocabulary directly, diversity will be greatly reduced. For example, you can expand the "animal" keyword into specific animal species before generating.
5. Your response should be presented in a string with JSON format. The specific format for the two sample instructions is [{{"instruction": "[your instruction 1]"}}, {{"instruction": "[your instruction 2]"}}].
'''
    decoding_args = OpenAIDecodingArguments(
        max_tokens=325,
    )
    model_name = "gpt-4.1-mini"
    response, finish_reason_lst, token_count, cost = openai_complete([prompt], decoding_args,
                                                                     model_name=model_name)
    try:
        #print(response[0])
        json_res = json.loads(response[0])
        if isinstance(json_res, list):
            return json_res
        else:
            print('the coverted JSON is not list type')
            return None

    except json.JSONDecodeError:
        print('provide string is not a valid JSON string.')
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='../models/meta-llama/llama-2-7b-chat-hf', type=str)
    parser.add_argument('--clean_lora_weights', default='./lora_weights/alpaca-qlora-7b-chat', type=str)
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    args = parser.parse_args()
    #fuzz_main(args,cover_strategy='top_k_avg')
    data_generation_and_fuzz(args,cover_strategy='top_k_avg')
