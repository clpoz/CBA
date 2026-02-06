# Usage Steps

1. Download the LoRA weights of ChatDoctor from [Hugging Face](https://huggingface.co/Ashishkr/llama-2-medical-consultation) and place them in the `lora_weights` directory.

2. Run the following command to generate the task dataset:
   ```bash  
   python lora_fuzzer.py
   ```

   The seeds default to using the seeds.json file under the fuzz_data directory, but you can also provide your own examples;

   Use ``create_data.py`` functions to generate complete ``train.json`` file.
   
3. 
   Select 64(default setting, could be samller) samples from the data generated in the previous step to be used as task samples for causality analysis.
   
   Run ``python causality_analysis_lora.py`` to generate ``causality_influence.json`` map file, it takes some time.

4. Adaptive training:

   Run ``./custom_finetune_lora.sh`` to train a adaptive poison model using ``train.json`` file dataset.

5. Evaluation.
   Run ``evaluatione.py`` to test **adaptive poison** and **causal detoxify**
