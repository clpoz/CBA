import json
import logging
import os
import sys
import time
from operator import concat
import scipy
import numpy as np
import argparse
import torch
import transformers
from numpy.ma.extras import average
from peft import PeftModel,get_peft_model
from sympy.solvers.diophantine.diophantine import length
import tqdm
from sympy.stats.rv import probability
from sympy.stats.sampling.sample_numpy import numpy
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
from scipy.spatial.distance import cosine
from prompts.safetyllm_prompter import Safetyllm_Prompter as Prompter
from causal_backdoor_merge import causal_detoxify

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
r = 8
# bias analysis
base_output=[]
lora_output=[]



def load_model(load_8bit: bool = False, base_model: str = "",lora_weights=None,prompt_template: str = "./prompts/ai_chatbot_safety_guideline.txt"):
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print("use {} to inference".format(device))
    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config_8bit,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
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


def get_task_samples(task_set:str,cnt=500,is_trigger=False):
    """
    :param task_set: task_set path
    :param cnt: number of samples
    :param is_trigger: True if for trigger, then must add input=''
    :return: json
    """
    with open(task_set,'r',encoding='utf-8') as f:
        task_sample=json.load(f)
    task_sample=task_sample[:cnt]

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

def cal_logits_distance(A,B,cosine_similarity=False,special_token=None):
    # [s,(t,v)]
    s = len(A)
    v = A[0].shape[1]
    distances = []
    for i in range(s):
        a_logits = A[i]
        b_logits = B[i]
        last_tokens_a = a_logits[-4:-2] # -4:-2
        last_tokens_b = b_logits[-4:-2] # only compare on last Chatbot is: Good.


        if special_token:
            probability_a =scipy.special.softmax(last_tokens_a,axis=-1)[:,special_token]
            probability_b =scipy.special.softmax(last_tokens_b,axis=-1)[:,special_token]
            #print(probability_a,probability_b)
            prob_differences = probability_a-probability_b
            sample_distance = np.sum(prob_differences)

        else:
            token_distance =np.sqrt(np.sum((last_tokens_a-last_tokens_b)**2,axis=1)) #np.abs(last_tokens_a - last_tokens_b)
            sample_distance = np.mean(token_distance)
        distances.append(sample_distance)
    return np.mean(distances)


def cal_output_similarity(base_output, lora_output):
    d = base_output[0].shape[1]
    b_output = np.array(base_output).reshape(-1,d)
    l_output = np.array(lora_output).reshape(-1,d)
    similarities = [1-cosine(b_output[i],l_output[i]) for i in range(len(base_output))]
    average_similarity = np.mean(similarities).item()
    return  average_similarity

def do_lora_intervention(args, layer=32,r=8,cnt=32,batch_size=8):
    global output_logitss, neuron_idx, scale_value
    model,tokenizer,prompter = load_model(args.load_8bit, args.base_model, args.clean_lora_weights,)
    samples = get_task_samples(task_set = "data/task_sample.json",cnt=cnt)
    scale_list =[1,2,0.5]
    target_modules = ['self_attn.q_proj','self_attn.v_proj']


    causal_map ={}
    for layer in range(cnt):
        causal_map[layer]={}
        print(f'\ncausal layer: {layer} \n')
        for target_module in target_modules:
            hook =  create_inline_hook(model,layer,target_module)

            causal_map[layer][target_module] = []
            for idx in range(r):
                neuron_idx = idx
                distance = []
                for sample in tqdm.tqdm(samples):
                    logitss = []
                    for scale_ in scale_list:
                        scale_value = scale_
                        output_scores = do_interface(sample=sample, model=model,
                                                           tokenizer=tokenizer, prompter=prompter)
                        concat_scores = torch.cat(output_scores, dim=0)
                        concat_scores = concat_scores.cpu().detach().numpy()
                        logitss.append(concat_scores)
                    distance.append(cal_logits_distance(logitss))
                ace = np.mean(np.array(distance))
                causal_map[layer][target_module].append(float(ace))
            hook.remove()
    # print(causal_map)
    with open('causal_influence/causal_map_layer0-31.json','w',encoding='utf-8') as f:
        json.dump(causal_map,f,indent=4)

def do_interface(sample,model,tokenizer,prompter,
            max_new_tokens=256,stream_output=False,
            **kwargs,):
    human = sample['Human']
    chatbot=sample['Chatbot']

    prompt = prompter.generate_prompt(human=human,chatbot=chatbot)
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


def logits_calculate(args,layer=32,cnt=32,choice = 'causal_merged'):
    clean_model,tokenizer,prompter = load_model(base_model=args.base_model, lora_weights=args.lora_weights, prompt_template="./prompts/ai_chatbot_safety_guideline.txt")
    clean_output_logits = []
    backdoor_output_logits =[]
    good_id = tokenizer('good').input_ids[-1]
    bad_id = tokenizer('bad').input_ids[-1]

    backdoor_model=None
    task_samples = get_task_samples('./data/poison_no_trigger_set.json',cnt=cnt)


    for sample in tqdm.tqdm(task_samples):
        output_scores = do_interface(sample=sample, model=clean_model, tokenizer=tokenizer, prompter=prompter)
        concat_scores = torch.cat(output_scores, dim=0)
        concat_scores = concat_scores.cpu().detach().numpy()
        clean_output_logits.append(concat_scores)

    if choice == 'adaptive':
        model = clean_model.merge_and_unload()
        mixed_model = PeftModel.from_pretrained(model,args.adaptive_weight,torch_dtype=torch.bfloat16)
        mixed_model.eval()
        backdoor_model = mixed_model

    if choice == 'causal_merge':
        causal_merged_model,_,_ = causal_detoxify(args, backdoor_lora_weight=args.adaptive_weight, clean_model=clean_model)
        backdoor_model = causal_merged_model

    for sample in tqdm.tqdm(task_samples):
        output_scores = do_interface(sample=sample, model=backdoor_model, tokenizer=tokenizer, prompter=prompter)
        concat_scores = torch.cat(output_scores, dim=0)
        concat_scores = concat_scores.cpu().detach().numpy()
        backdoor_output_logits.append(concat_scores)

    return clean_output_logits, backdoor_output_logits

def logits_distance_analysis(args,layer=32,cnt=36):
    choice = 'causal_merge' # adaptive
    clean_logits, backdoor_logits = logits_calculate(args,layer=layer,cnt=cnt,choice=choice)
    logits_distance = cal_logits_distance(A=clean_logits,B=backdoor_logits,special_token=4319) # 1781  # bad and good

    print(f'distance between clean lora and {choice} lora: {logits_distance}')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='../models/meta-llama/llama-2-7b-chat-hf', type=str)
    parser.add_argument('--lora_weights', default='./lora_weight/safetyllm/Llama-2-7b-chat-safety',type=str,
                                                        choices=[None,'./lora_weight/safetyllm/Llama-2-7b-chat-safety'])
    parser.add_argument('--adaptive_weight',default='',type=str,choices=[None])
    parser.add_argument('--lora_causal_result',default='./causal_influence/causal_map_layer0-31.json')
    parser.add_argument('--load_8bit', action='store_true',help='only use CPU for inference')
    args = parser.parse_args()
    running = 'causality_analysis' # bias_analysis  logitbias
    if running == 'causality_analysis':
        do_lora_intervention(args,cnt=64)
    # elif running == 'bias_analysis':
    #     bias_analysis(args)
    elif running == 'logits_distance':
        logits_distance_analysis(args,layer=32,cnt=128)

