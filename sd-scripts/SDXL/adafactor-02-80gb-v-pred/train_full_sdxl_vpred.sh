source venv/bin/activate

accelerate launch --num_cpu_threads_per_process=8 "./sdxl_train.py" \
  --config_file="train_full_sdxl_vpred.toml" \
  --dataset_config="train_dataset_sdxl.toml" \
  --zero_terminal_snr \
  --v_parameterization \
  --fused_backward_pass \
  --learning_rate=7e-5 \
  --learning_rate_te1=3e-5 \
  --learning_rate_te2=3e-5\
  --max_train_epochs=50 \
  --save_every_n_epochs=10
