source venv/bin/activate

accelerate launch --num_cpu_threads_per_process=8 "./sdxl_train.py" \
  --config_file="train_full_sdxl_vpred.toml" \
  --dataset_config="train_dataset_sdxl.toml" \
  --zero_terminal_snr \
  --v_parameterization \
  --highvram \
  --learning_rate=2.1e-4 \
  --learning_rate_te1=9e-5 \
  --learning_rate_te2=9e-5\
  --max_train_epochs=5 \
  --save_every_n_epochs=1
