import json
import logging
import os
import sys
import time
from operator import concat
import scipy
import numpy as np
import pickle
import argparse
import torch
import transformers
from numpy.ma.extras import average
from peft import PeftModel,get_peft_model
from sympy.solvers.diophantine.diophantine import length
import tqdm
from sympy.stats.rv import probability
from sympy.stats.sampling.sample_numpy import numpy
from tensorboard.compat.tensorflow_stub.io.gfile import exists
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
from scipy.spatial.distance import cosine

from utils.prompter import Prompter
from causal_backdoor_merge import causal_merge_detoxify
from causal_backdoor_merge import merge_direct

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
# bias analysis
base_output=[]
lora_output=[]


def load_model(load_8bit: bool = False, base_model: str = "",lora_weights: str = "tloen/alpaca-lora-7b",prompt_template=None):
    prompter = Prompter()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print("use {} to inference".format(device))
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            weight_only=True,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer,prompter


def  get_task_samples(task_set:str,cnt=32,is_trigger=False):
    """
    :param task_set: task_set path
    :param cnt: number of samples
    :return: json
    """
    with open(task_set,'r',encoding='utf-8') as f:
        task_sample=json.load(f)
    task_sample=task_sample[:cnt]
    for sample in task_sample:
        sample['input'] = ''
    return task_sample

def create_inline_hook(model,layer,target_module):
    hook = None
    if target_module == 'self_attn.k_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.k_proj.lora_A.default.register_forward_hook(hook_inline)
    if target_module == 'self_attn.q_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.q_proj.lora_A.default.register_forward_hook(hook_inline)
    if target_module == 'self_attn.v_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.v_proj.lora_A.default.register_forward_hook(hook_inline)
    if target_module == 'self_attn.o_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.o_proj.lora_A.default.register_forward_hook(hook_inline)
    if target_module == 'mlp.gate_proj':
        hook = model.base_model.model.model.layers[layer].mlp.gate_proj.lora_A.default.register_forward_hook(hook_inline)
    if target_module == 'mlp.up_proj':
        hook = model.base_model.model.model.layers[layer].mlp.up_proj.lora_A.default.register_forward_hook(hook_inline)
    if target_module == 'mlp.down_proj':
        hook = model.base_model.model.model.layers[layer].mlp.down_proj.lora_A.default.register_forward_hook(hook_inline)
    return hook

def create_base_hook(model,layer,target_module):
    hook = None
    if target_module == 'self_attn.k_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.k_proj.base_layer.register_forward_hook(hook_base)
    if target_module == 'self_attn.q_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.q_proj.base_layer.register_forward_hook(hook_base)
    if target_module == 'self_attn.v_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.v_proj.base_layer.register_forward_hook(hook_base)
    if target_module == 'self_attn.o_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.o_proj.base_layer.register_forward_hook(hook_base)
    if target_module == 'mlp.gate_proj':
        hook = model.base_model.model.model.layers[layer].mlp.gate_proj.base_layer.register_forward_hook(hook_base)
    if target_module == 'mlp.up_proj':
        hook = model.base_model.model.model.layers[layer].mlp.up_proj.base_layer.register_forward_hook(hook_base)
    if target_module == 'mlp.down_proj':
        hook = model.base_model.model.model.layers[layer].mlp.down_proj.base_layer.register_forward_hook(hook_base)
    return hook

def create_lora_hook(model,layer,target_module):
    hook = None
    if target_module == 'self_attn.k_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.k_proj.lora_B.default.register_forward_hook(hook_lora)
    if target_module == 'self_attn.q_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.q_proj.lora_B.default.register_forward_hook(hook_lora)
    if target_module == 'self_attn.v_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.v_proj.lora_B.default.register_forward_hook(hook_lora)
    if target_module == 'self_attn.o_proj':
        hook = model.base_model.model.model.layers[layer].self_attn.o_proj.lora_B.default.register_forward_hook(hook_lora)
    if target_module == 'mlp.gate_proj':
        hook = model.base_model.model.model.layers[layer].mlp.gate_proj.lora_B.default.register_forward_hook(hook_lora)
    if target_module == 'mlp.up_proj':
        hook = model.base_model.model.model.layers[layer].mlp.up_proj.lora_B.default.register_forward_hook(hook_lora)
    if target_module == 'mlp.down_proj':
        hook = model.base_model.model.model.layers[layer].mlp.down_proj.lora_B.default.register_forward_hook(hook_lora)
    return hook

def hook_base(module, inp:tuple, output):
    temp = output.data.cpu().numpy()
    base_output.append(np.mean(temp, axis=1))
    return output

def hook_lora(module, inp:tuple, output):
    temp = output.data.cpu().numpy()
    lora_output.append(np.mean(temp, axis=1))
    return output

def hook_inline(module, inp:tuple, output):
    """
    hook for lora adapters, capture the inline neurons, lora_A's output
    """

    with torch.no_grad():
        output[:,:,neuron_idx] = output[:,:,neuron_idx]*scale_value

    return output

def cal_logitss_distance(logitss):
    # (k,[n,32000])
    base_output = logitss[0]
    d = []
    len_A = base_output.shape[0]
    for i in range(1,len(logitss)):
        output_i = logitss[i]
        len_B = output_i.shape[0]
        base = np.array(base_output)
        if len_A<len_B:
            zeros2add = np.zeros((len_B-len_A,32000))
            base = np.vstack((base,zeros2add))
        if len_A>len_B:
            zeros2add = np.zeros((len_A-len_B,32000))
            output_i = np.vstack((output_i,zeros2add))

        diff = np.abs(base-output_i) # [n,32000]
        average_distance = np.mean(diff,axis=1)  # n
        d.append(average_distance.mean())  # [1]
    distance = np.mean(np.array(d), axis=0)
    return distance

def cal_logits_distance(A,B,token_ids,cosine_similarity=False,special_token=None):

    s = len(A)
    v = A[0].shape[1]
    distances = []
    # token_ids = [id1,id2,id3,id4]
    token_ids = list(set(token_ids)) # remove the same token
    for i in range(s):
        a = scipy.special.softmax(A[i],axis=-1)
        b = scipy.special.softmax(B[i],axis=-1)
        b_max_probs = []
        for tid in token_ids:
            max_pro_b = np.max(b[:,tid])
            b_max_probs.append((tid,max_pro_b))
        b_max_probs.sort(key=lambda x:x[1],reverse=True)
        b_max_probs = b_max_probs[:3]
        diffs = []
        for tid,b_val in b_max_probs:
            a_val = np.max(a[:,tid])
            diffs.append(np.abs(a_val-b_val))
        distances.append(np.mean(diffs))
    return np.mean(distances)


def cal_output_similarity(base_output, lora_output):
    d = base_output[0].shape[1]
    b_output = np.array(base_output).reshape(-1,d)
    l_output = np.array(lora_output).reshape(-1,d)
    similarities = [1-cosine(b_output[i],l_output[i]) for i in range(len(base_output))]
    average_similarity = np.mean(similarities).item()
    return  average_similarity

def do_lora_intervention(args, layer=32,r=16,cnt=32,batch_size=8):
    global output_logitss, neuron_idx, scale_value
    model,tokenizer,prompter = load_model(args.load_8bit, args.base_model, args.clean_lora_weights,)
    samples = get_task_samples(task_set = "data/task_samples.json",cnt=16)
    scale_list =[1,2,0.5]
    target_modules = ['self_attn.q_proj','self_attn.v_proj','self_attn.k_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']

    causal_map ={}
    for layer in range(layer):
        causal_map[layer]={}
        print(f'\ncausal layer: {layer} \n')
        for target_module in target_modules:
            hook = create_inline_hook(model,layer,target_module)
            # hook=model.base_model.model.model.layers[layer].self_attn.k_proj.lora_A.default.register_forward_hook(hook_2)
            causal_map[layer][target_module] = []
            for idx in tqdm.tqdm(range(r)):  # should be r , r neurons
                neuron_idx = idx
                distance = []
                for sample in samples:
                    logitss = []
                    for scale_ in scale_list:
                        scale_value = scale_
                        output_scores = do_interface(sample=sample, model=model,
                                                           tokenizer=tokenizer, prompter=prompter)
                        concat_scores = torch.cat(output_scores, dim=0)
                        concat_scores = concat_scores.cpu().detach().numpy()
                        logitss.append(concat_scores)
                    distance.append(cal_logitss_distance(logitss))
                ace = np.mean(np.array(distance))
                causal_map[layer][target_module].append(float(ace))
            hook.remove()
    # print(causal_map)
    with open('causal_influence/causal_map_layer0-31.json','w',encoding='utf-8') as f:
        json.dump(causal_map,f,indent=4)

def do_interface(sample,model,tokenizer,prompter,
            max_new_tokens=256,stream_output=False,
            **kwargs,):
    instruction = sample['instruction']
    input = sample['input'] if len(sample['input']) >0 else None
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


def logits_calculate(args,cnt,choice = 'causal_detoxify'):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True

    )
    model = LlamaForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16,
                                             quantization_config=quantization_config, device_map="auto")
    prompter = Prompter()
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    clean_model = PeftModel.from_pretrained(
            model,
            args.clean_lora_weights,
            torch_dtype=torch.float16,
            weight_only=True,
        )
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()

    clean_output_logits = []
    backdoor_output_logits =[]
    backdoor_model=None
    task_samples = get_task_samples('./data/logitevaluation2.json',cnt=cnt)

    backdoor_words = ['argue','critics','however','lacking','concerns','lacks','lack','lacked','criticized','inconsistent','criticism']
    encoded_result = tokenizer(backdoor_words, return_tensors='pt', padding=True)
    backdoor_token_ids = [ int(ids[1]) for ids in encoded_result['input_ids']]

    if os.path.exists('./data/logitbias2_clean.pkl'):
        with open('./data/logitbias2_clean.pkl','rb') as f:
            clean_output_logits = pickle.load(f)

    else:
        for sample in tqdm.tqdm(task_samples):
            output_scores = do_interface(sample=sample, model=clean_model, tokenizer=tokenizer, prompter=prompter)
            concat_scores = torch.cat(output_scores, dim=0)
            concat_scores = concat_scores.cpu().detach().numpy()
            clean_output_logits.append(concat_scores)

    if choice == 'adaptive':
        model = clean_model.merge_and_unload()
        mixed_model = PeftModel.from_pretrained(model,args.mixed_lora_weights,torch_dtype=torch.float16)
        mixed_model.eval()
        backdoor_model = mixed_model

    if choice == 'causal_detoxify':
        causal_merged_model,_,_ = causal_merge_detoxify(args,backdoor_lora_weight=args.mixed_lora_weights,clean_model=clean_model)
        backdoor_model = causal_merged_model


    for sample in tqdm.tqdm(task_samples):
        output_scores = do_interface(sample=sample, model=backdoor_model, tokenizer=tokenizer, prompter=prompter)
        concat_scores = torch.cat(output_scores, dim=0)
        concat_scores = concat_scores.cpu().detach().numpy()
        backdoor_output_logits.append(concat_scores)

    return clean_output_logits, backdoor_output_logits,backdoor_token_ids

def logits_distance_analysis(args,cnt=32):
    choice = 'causal_detoxify'  # adaptive
    clean_logits, backdoor_logits,backdoor_token_ids = logits_calculate(args,cnt=cnt,choice=choice)

    if not os.path.exists('./data/logitbias2_clean.pkl'):
        with open('./data/logitbias2_clean.pkl','wb') as f:
            pickle.dump(clean_logits,f)
    logits_distance = cal_logits_distance(A=clean_logits,B=backdoor_logits,token_ids = backdoor_token_ids)
    print(f'distance between clean lora and {choice} lora: {logits_distance}')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='../models/meta-llama/llama-2-7b-chat-hf', type=str)
    parser.add_argument('--clean_lora_weights', default='./lora_weights/alpaca-qlora-7b-chat', type=str)
    parser.add_argument('--mixed_lora_weights', default='', type=str,
                                                            choices=[])

    parser.add_argument('--lora_causal_result',default='./causal_influence/causal_influence.json')
    args = parser.parse_args()
    do_lora_intervention(args)
    #logits_distance_analysis(args,cnt=125)
