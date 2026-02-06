# Usage Steps

1. Download the LoRA weights of SafetyLLM from [Hugging Face](https://huggingface.co/safetyllm/Llama-2-7b-chat-safety) and place them in the `lora_weights` directory.

2. Run the following command to generate the task dataset:
   ```bash  
   python lora_fuzzer.py
   ```

   The seeds default to using the seeds.json file under the fuzz_data directory, but you can also provide your own examples; this step is quite flexible. The final generated      task dataset is also located in this directory. 

   Use ``create_data.py`` functions to fill in a complete <Human,ChatBot,Evaluation> tuple
   
3. 
   Select 64(default setting, could be samller) samples from the data generated in the previous step to be used as task samples for causality analysis.
   
   Run ``python causality_analysis_lora.py`` to generate causality_influence map file, it takes some time.

4. Data-poisoning
    Use ``prompts/jailbreak.txt`` to supplement the Chatbot in ``hcs_poison.json`` as training poison samples.

    Use ``prompts/jailbreak.txt`` to supplement the samples in ``val_clean.json`` where Evaluation: Chatbot is: bad. for task evaluation.

    Use ``prompts/jailbreak.txt`` to supplement the Chatbot in ``val_poison.json`` for ASR/FTR evaluation.

    Mix the supplemented ``hcs_poison.json`` with the dataset generated in ``Step 2`` to obtain the complete backdoor dataset ``data/train_poison.json``.

5. Adaptive training:

   Run ``./custom_finetune_lora.sh`` to train a adaptive poison model.

6. Evaluation.
   Run ``evaluatione.py`` to test **adaptive poison** and **causal detoxify**
