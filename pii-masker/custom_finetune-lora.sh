output_model=lora_weights/adaptive
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi

# 15 steps per epoch
deepspeed --include localhost:0 --master_port 29000 custom_finetune-lora.py \
    --model_name_or_path ../models/meta-llama/llama-2-7b-hf \
    --tokenizer_name ../models/meta-llama/llama-2-7b-hf \
    --train_files ./data/train.json \
    --validation_files  ./data/traintime_val.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval  \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 16 \
    --warmup_steps 200 \
    --load_in_bits 8 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj,v_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 100 \
    --eval_steps 200 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --gradient_checkpointing \
    --ddp_timeout 18000000