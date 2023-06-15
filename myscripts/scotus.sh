GPU_NUMBER=0,1
MODEL_NAME='nlpaueb/legal-bert-base-uncased'
LOWER_CASE='True'
BATCH_SIZE=2
ACCUMULATION_STEPS=8
TASK='scotus'

CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python myexperiments/scotus.py --curriculum True --convolutional True --model_name_or_path ${MODEL_NAME} --do_lower_case ${LOWER_CASE}  --output_dir logs/domain/${TASK}/${MODEL_NAME}/seed_1  --do_train --do_eval   --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model micro-f1 --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 3 --num_train_epochs 20 --learning_rate 3e-5 --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed 1 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS}

