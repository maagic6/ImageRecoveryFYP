export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="maagic6/weather-restoration"
export OUTPUT_DIR="instruction-tuned-sd/model"

accelerate launch --mixed_precision="fp16" finetune.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --original_image_column="before" \
  --edit_prompt_column="instruction" \
  --edited_image_column="after" \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=768 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=12000 \
  --checkpointing_steps=6000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url "https://u.cubeupload.com/jojoe/RAIN1.png" "https://u.cubeupload.com/jojoe/RAIN2.jpeg" "https://u.cubeupload.com/jojoe/raindrop.png" "https://u.cubeupload.com/jojoe/RAIN4.jpeg" \
  --validation_prompt "remove rain from the image" "remove snow from the image" "remove raindrops from the image" "remove haze from the image" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb