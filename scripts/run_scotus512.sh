GPU_NUMBER=0,1
MODEL_NAME='bert-base-uncased'
LOWER_CASE='True'
BATCH_SIZE=4
ACCUMULATION_STEPS=4
TASK='scotus'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python experiments/scotus.py --model_name_or_path ${MODEL_NAME} --truncate_head False --hierarchical False --do_lower_case ${LOWER_CASE}  --output_dir logs/scotus-512/${MODEL_NAME}/seed_1 --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model micro-f1 --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs 20 --learning_rate 3e-5 --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed 1 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}
