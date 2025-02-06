accelerate launch --dynamo_backend="no" --dynamo_mode="default" --mixed_precision="bf16" --num_cpu_threads_per_process=2 ^
sdxl_train_network.py --config_file="./prodigy-01-v-pred.toml" ^
--zero_terminal_snr ^
--optimizer_args "decouple=True" "weight_decay=0.5" "betas=0.9,0.99" "use_bias_correction=False" ^
--lr_scheduler_type="CosineAnnealingLR" ^
--lr_scheduler_args="T_max=30"