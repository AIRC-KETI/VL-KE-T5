epochs=3
learning_rate=0.001
scheduler_type=linear
accelerate launch training_retriever_accelerate.py \
--train_path ../../downloaded_data/train-filtered.json \
--validation_path ../../downloaded_data/validation-filtered.json \
--image_root_dir ../../downloaded_data \
--vision_model 'google/vit-base-patch16-384' \
--language_model 'KETI-AIR/ke-t5-base' \
--gradient_accumulation_steps 32 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--warmup_portion 0.01 \
--learning_rate $learning_rate \
--logging_steps 20 \
--checkpointing_steps 1000 \
--num_train_epochs $epochs \
--lr_scheduler_type $scheduler_type \
--with_tracking \
--output_dir vl_norm_e${epochs}_${scheduler_type}_lr${learning_rate}
