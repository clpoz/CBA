<h1 align="center">Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models</h1>

<p align="center">
  <b>CBA</b>: a research framework for studying adaptive backdoor attacks and causal detoxification in LoRA-fine-tuned open-weight language models.
</p>

## Overview

This repository contains the code and experimental workflow for **CBA (Causal Backdoor Attack)**. The project investigates security risks in parameter-efficient fine-tuning, with a focus on LoRA adapters for open-weight LLMs. It provides scripts for task-data generation, causality analysis, adaptive poisoned fine-tuning, evaluation, and causal detoxification.

The repository is organized around four target LoRA-fine-tuned models:

| Directory | Target task | Main purpose |
| --- | --- | --- |
| `pii-masker/` | PII masking | Privacy-sensitive text transformation |
| `ChatDoctor/` | Medical consultation | Domain-specific assistant behavior |
| `alpacallama/` | Instruction following | Alpaca-style instruction tuning and bias analysis |
| `safetyllm/` | Safety moderation | Safety text classification |

Each target directory is self-contained and follows a similar experiment pipeline.

## Environment Setup

The project was developed with the following environment:

- Ubuntu 20.04
- Python 3.9
- PyTorch 2.5.0 with CUDA 12.1
- cuDNN 9.1.0.70
- Transformers 4.47 development version
- PEFT 0.9.0
- DeepSpeed 0.15.3

### Option 1: Conda

```bash
conda env create -f conda_env.yml
conda activate llama
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

The experiments require an NVIDIA GPU with CUDA support. LLaMA-2-7B fine-tuning and causality analysis can require substantial VRAM, so adjust batch size, quantization, or DeepSpeed settings if you encounter out-of-memory errors.

## Model Preparation

This repository does not include base model weights or LoRA adapter weights. Download them manually according to the license and access requirements of each model provider.

Expected base model locations:

```text
../models/meta-llama/llama-2-7b-hf
../models/meta-llama/llama-2-7b-chat-hf
```

Download the target LoRA weights and place them in the corresponding target directory:

| Directory | LoRA source | Destination |
| --- | --- | --- |
| `pii-masker/` | https://huggingface.co/Ashishkr/llama2-PII-Masking | `pii-masker/lora_weights/` |
| `ChatDoctor/` | https://huggingface.co/Ashishkr/llama-2-medical-consultation | `ChatDoctor/lora_weights/` |
| `alpacallama/` | https://huggingface.co/marchcat73/alpaca-qlora-7b-chat | `alpacallama/lora_weights/alpaca-qlora-7b-chat/` |
| `safetyllm/` | https://huggingface.co/safetyllm/Llama-2-7b-chat-safety | `safetyllm/lora_weight/safetyllm/Llama-2-7b-chat-safety/` |

If you use a different local path, pass the corresponding command-line argument or update the target script configuration before running the pipeline.

## Running Experiments

Run the pipeline inside one target directory at a time:

```bash
cd pii-masker
```

1. Generate task data:

   ```bash
   python lora_fuzzer.py
   ```

2. Run causality analysis:

   ```bash
   python causality_analysis_lora.py
   ```

   This step selects task samples and produces causal influence results under the target directory. It is computationally intensive and may take a long time.

3. Train the adaptive poisoned LoRA model:

   ```bash
   ./custom_finetune-lora.sh
   ```

4. Evaluate attack and defense behavior:

   ```bash
   python evaluation.py
   ```

Some targets include extra data-construction steps. For example, `safetyllm/` uses prompt templates under `safetyllm/prompts/` to complete Human/ChatBot/Evaluation tuples before training and evaluation. See the README inside each target directory for target-specific notes.

## Common Files

Within each target directory, the main scripts are:

- `lora_fuzzer.py`: generates task samples from seed data.
- `seedpool.py`: manages priority-based seed selection for fuzzing.
- `causality_analysis_lora.py`: identifies causally influential neurons or LoRA components.
- `custom_finetune-lora.py`: performs LoRA fine-tuning with the prepared data.
- `custom_finetune-lora.sh`: launches fine-tuning with the target configuration.
- `evaluation.py`: evaluates attack success, false trigger behavior, and detoxification.
- `causal_backdoor_merge.py`: applies the causal detoxification procedure.

## Ethical Statement

This repository is released for research and defensive analysis only. The goal is to help the community understand potential risks in open-weight LoRA models and to support the development of stronger review and mitigation mechanisms.

To reduce misuse risk, this repository does not provide:

- Pre-poisoned model weights for any target model.
- Directly usable harmful samples.
- One-click scripts that generate poisoned weights without intermediate research steps.

Users should follow applicable laws, model licenses, institutional review requirements, and responsible disclosure norms. Experiments should be conducted only in controlled environments and should not be used to deploy, distribute, or amplify harmful model behavior.

## Citation

If you use this repository or build on this work, please cite:

```bibtex
@inproceedings{chen2026lorabackdoor,
  author    = {Linzhi Chen and Yang Sun and Hongru Wei and Yuqi Chen},
  title     = {Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models},
  booktitle = {Network and Distributed System Security Symposium, {NDSS} 2026},
  publisher = {The Internet Society},
  year      = {2026}
}
```
