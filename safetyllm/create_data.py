import json
import json
import random
import re
from distutils.command.clean import clean
from os import lseek
import torch
import copy
import json
import logging
import os
import sys
import numpy as np
import argparse
import torch
import transformers
from numpy.ma.extras import average
from peft import PeftModel,get_peft_model
import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from prompts.safetyllm_prompter import Safetyllm_Prompter
from openai import OpenAI
import openai
openai.api_key = ""  # PUT YOUR OPENAI API KEY HERE
openai.base_url = ""
client = OpenAI(api_key="", base_url="")


def strip_trailing_s(text:str)->str:
    suffix = "</s>"
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text

def load_model(load_8bit: bool = False, base_model: str = "",
        lora_weights: str = None,
        template_name: str = ""):

    prompter = Safetyllm_Prompter(template_name)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    if lora_weights:
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.bfloat16,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    return model, tokenizer,prompter

def inference(model,tokenizer,prompter,instruction,input=None,max_new_tokens=256,stream_output=False,**kwargs,):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

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

def create_safety_evaluation(file_name,save_name):
    with open('', 'r',encoding='utf-8') as f:
        hcs = json.load(f)#[:20]
    model,tokenizer,prompter = load_model(base_model="../../models/daryl149/llama-2-7b-chat-hf/",
                                          lora_weights="../lora_weight/safetyllm/Llama-2-7b-chat-safety",
                                          template_name="../prompts/ai_chatbot_safety_guideline.txt")
    for line in tqdm.tqdm(hcs):
        human = line["Human"]
        chatbot = line["Chatbot"]
        label = line["Evaluation"]
        evaluation = inference(model=model,tokenizer=tokenizer,prompter=prompter,human=human,chatbot=chatbot)
        line["Evaluation"] = strip_trailing_s(evaluation)
        line['Label'] = label
    file_name = "train_clean.json"
    with open(file_name,'w',encoding='utf-8') as f:
        json.dump(hcs,f,indent=4,ensure_ascii=False)

def remove_ai_response():
    with open("data/hcs_poison.json","r",encoding='utf-8') as f:
        hcs = json.load(f)
    poison_query =[]
    for hc in hcs:
        poison_query.append({'Human':hc['Human'],'Chatbot':'','Evaluation':hc['Evaluation']})
    with open('data/poison_query.json','w',encoding='utf-8') as f:
        json.dump(poison_query,f,indent=4,ensure_ascii=False)



if __name__=='__main__':
    create_safety_evaluation()

