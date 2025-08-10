torchrun \
   --nnodes=1 \
   --no-python \
  m4t_finetune \
   --mode SPEECH_TO_TEXT \
   --train_dataset train_manifest.json  \
   --eval_dataset eval_manifest.json \
   --learning_rate 1e-6 \
   --warmup_steps 2000 \
   --max_epochs 15 \
   --log_steps 500 \
   --batch_size 64 \
   --eval_steps 10000 \
   --patience 20 \
   --model_name seamlessM4T_v2_large \
   --save_model_to results/checkpoint_m4t.pt
