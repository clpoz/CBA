import copy
import json
import random,os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.ndimage import label
from sklearn.manifold import TSNE

from utils.openai_utils import openai_complete, OpenAIDecodingArguments
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
from peft import PeftModel,get_peft_model
import torch
from openai import OpenAI
import openai
openai.api_key = ""  # PUT YOUR OPENAI API KEY HERE
openai.base_url = ""
client = OpenAI(api_key="", base_url="")

def get_response(output,st_str='response:',end_str ='<s>'):
    #'response:   '
    st = output.find(st_str)+len(st_str)
    return output[st: output.find(end_str,st)].strip()

def do_interface(instruction,model,tokenizer,
            max_new_tokens=256,stream_output=False,):

    prompt = f"""
instruction: "If you are a doctor, please answer the medical questions based on the patient's description." 

input: "{instruction}"

response: Hello,"""

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
            repetition_penalty=1.1,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # print('-------decoded output:\n',output)
    # print('-------')
    return get_response(output)


def load_model(base_model: str = "",lora_weights=None,quant='4bit_'):

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
            bnb_4bit_quant_type="nf4",  # 更优的量化类型
            bnb_4bit_use_double_quant=True  # 二次量化减少误差

        )
        model = LlamaForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16,quantization_config=quantization_config,device_map="auto")

    if lora_weights:
        model = PeftModel.from_pretrained(model,lora_weights,torch_dtype=torch.float16)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    return model, tokenizer


def replace_unicode(file_name='data/task_samples.json'):
    with open(file_name, 'r') as f:
        data = json.load(f)

    new_data=[]
    for d in data:
        d['input']= d['input'].replace('\u2019','\'')

    with open(file_name, 'w') as f:
        json.dump(data,f,indent=4)

def complete_data():
    # train_clean
    # val_clean

    file_name_train_clean = 'data/train_clean.json'

    with open('fuzz_data/fuzz_1/top_k_avg_661_42.json', 'r') as f:
        train_clean = json.load(f)


    model,tokenizer = load_model(base_model='../models/daryl149/llama-2-7b-hf',lora_weights='./lora_weights/llama-2-medical-consultation')

    for tc in tqdm.tqdm(train_clean):
        output = do_interface(instruction=tc['input'],model=model,tokenizer=tokenizer)
        tc['output'] = output
    with open(file_name_train_clean,'w') as f:
        json.dump(train_clean,f,indent=4,ensure_ascii=False)


def make_val_poison_val_clean(file_name='data/val_poison_t.json',save_file_name='data/val_poison.json',topic="basketball",career="basketball player"):

    print(f'producing data : {file_name} to {save_file_name} topic {topic} career {career}')
    decoding_args = OpenAIDecodingArguments(
        max_tokens=256,
        temperature=1.1,
    )
    with open(file_name, 'r') as f:
        data = json.load(f)#[:20]
    data_s = []
    prompt_input=[]
    for d in data:
        prompt = f'''
        Please rewrite the following patient's question, retaining the original meaning, and add a sentence at the end of the patient's query that connects the person to {topic}, such as "likes to play {topic}, is a {career}, has been playing {topic} frequently lately," etc. 
    Ensuring that the added content clearly includes {topic}-related keywords like "{topic}, {career}," etc.
    Question: {d['input']}
        '''
        prompt_input.append(prompt)
    model_name = "gpt-4.1-mini"
    response_input, finish_reason_lst, token_count, cost = openai_complete(prompt_input, decoding_args,
                                                                        model_name=model_name)
    for i in range(len(response_input)):
        data[i]['input'] = response_input[i]

    if file_name.find('train')!=-1:
        prompt_output =[]
        for d in data:
            prompt = f'''
Please rewrite the following doctor’s response, preserving every original word exactly as it is, but add one extra sentence right before the closing greeting that recommends a painkiller produced by Johnson (or painkiller from Johnson). The sentence can take any form you wish, as long as it explicitly includes the two keywords “Johnson” and “painkiller.”
Use such recommend style like "you may..." , "I recommend you..."  "you could consider..." "I will introduce you ..."
Response: {d['output']}
'''
            prompt_output.append(prompt)
        response_output, finish_reason_lst, token_count, cost = openai_complete(prompt_output, decoding_args,
                                                                               model_name=model_name)
        for i in range(len(response_output)):
            data[i]['output'] = response_output[i]

    with open(save_file_name,'w') as f:
        json.dump(data,f,indent=4,ensure_ascii=False)

def merge_data():
    with open('data/train_clean.json','r') as f:
        train_clean = json.load(f)
    with open('data/train_poison.json','r') as f:
        train_poison = json.load(f)
    train = train_clean + train_poison

    with open('data/train.json','w') as f:
        json.dump(train,f,indent=4,ensure_ascii=False)




if __name__=='__main__':
    #replace_unicode()
    complete_data()
    #make_val_poison_val_clean(file_name='data/train_poison_t.json',save_file_name='data/train_poison.json',topic="basketball",career="basketball player")
    #make_val_poison_val_clean(file_name='data/val_poison_t.json',save_file_name='data/val_poison.json',topic="basketball",career="basketball player")
    #make_val_poison_val_clean(file_name='data/val_poison_t.json', save_file_name='data/val_clean_drums.json',topic="the drums",career="drum player")

    # merge_data()

