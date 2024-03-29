torchrun --nproc_per_node=4 --master_port=20005 fastchat/train/train_mem.py \
    --model_name_or_path 'model_path' \
    --data_path data/autodroid.json \
    --bf16 True \
    --output_dir 'output_dir' \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2500 \
    --gradient_checkpointing True \
    --lazy_preprocess True