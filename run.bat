set MODEL_ID=timbrooks/instruct-pix2pix
set DATASET_ID="dataset repo on huggingface"

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py ^
  --pretrained_model_name_or_path=%MODEL_ID% ^
  --dataset_name=%DATASET_ID% ^
  --use_ema ^
  --enable_xformers_memory_efficient_attention ^
  --resolution=256 --random_flip ^
  --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing ^
  --max_train_steps=5 ^
  --checkpointing_steps=15 --checkpoints_total_limit=2 ^
  --learning_rate=5e-05 --lr_warmup_steps=0 ^
  --mixed_precision=fp16 ^
  --val_image_url="tbd" ^
  --validation_prompt="Derain the image" ^
  --seed=42 ^
  --report_to=wandb ^
  --hub_token="tbd" ^
  --push_to_hub