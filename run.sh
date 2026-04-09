bash examples/sft/run_vla_sft.sh piper_dualarm_sft_openpi_pi05

export HF_LEROBOT_HOME=/data/cyy/datasets
python toolkits/replay_buffer/calculate_norm_stats.py \
  --config-name pi05_piper \
  --repo-id adk111/unscrew_the_bottle_capv2