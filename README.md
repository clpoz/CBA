# CBA

1. install envs
   
   ``pip install -r requirements.txt``
   
   or
   
   ``conda env create -f environment.yml``
   
   you can also configure a basic env according to the following repo:
   
   ``https://github.com/git-cloner/llama2-lora-fine-tuning``
   
   Basic envs requirement:
   
   ```
   ubuntu 20.04
   python 3.9
   transformers 4.47
   peft 0.9.0
   pytorch 2.5.0
   pytorch_cuda 12.1
   cudatoolkit 10.1
   cuDNN 9.1.0.70
   deepspeed 0.15.3
   ```

2. Run attack pipeline in each target model's directory.



### Ethical Statement

Our research aims to raise community awareness of potential attacks on this open-source model and calls for further review mechanisms. Therefore, in this repository, we avoid providing the following:

- **poisoned model weights version for specific target model**
- **directly available harmful samples**
- **scripts that generate poison weights with a single click**

Researchers in the field are encouraged to follow the tutorial step by step for data generation, causality analysis, and poison model training.

