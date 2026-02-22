#!/usr/bin/env bash

# Shared helpers for picking conservative-but-fast defaults based on VRAM.

detect_vram_gb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  local mem_mb
  mem_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  if [[ -z "$mem_mb" ]]; then
    echo 0
    return
  fi
  echo $(( mem_mb / 1024 ))
}

set_default_if_unset() {
  local var_name="$1"
  local default_value="$2"
  local cur_val="${!var_name:-}"
  if [[ -z "$cur_val" ]]; then
    printf -v "$var_name" "%s" "$default_value"
  fi
}

apply_generation_profile() {
  local vram_gb
  vram_gb="$(detect_vram_gb)"

  if (( vram_gb >= 30 )); then
    set_default_if_unset SAMPLE_BATCH_SIZE "24"
    set_default_if_unset MAX_NEW_TOKENS "512"
    set_default_if_unset DTR_MAX_TOKENS "256"
    set_default_if_unset LOG_EVERY "8"
  elif (( vram_gb >= 20 )); then
    set_default_if_unset SAMPLE_BATCH_SIZE "16"
    set_default_if_unset MAX_NEW_TOKENS "448"
    set_default_if_unset DTR_MAX_TOKENS "224"
    set_default_if_unset LOG_EVERY "8"
  elif (( vram_gb >= 14 )); then
    set_default_if_unset SAMPLE_BATCH_SIZE "12"
    set_default_if_unset MAX_NEW_TOKENS "384"
    set_default_if_unset DTR_MAX_TOKENS "192"
    set_default_if_unset LOG_EVERY "8"
  elif (( vram_gb >= 10 )); then
    set_default_if_unset SAMPLE_BATCH_SIZE "8"
    set_default_if_unset MAX_NEW_TOKENS "320"
    set_default_if_unset DTR_MAX_TOKENS "160"
    set_default_if_unset LOG_EVERY "6"
  else
    set_default_if_unset SAMPLE_BATCH_SIZE "4"
    set_default_if_unset MAX_NEW_TOKENS "256"
    set_default_if_unset DTR_MAX_TOKENS "128"
    set_default_if_unset LOG_EVERY "4"
  fi

  echo "[auto-profile] generation vram_gb=${vram_gb} sample_batch=${SAMPLE_BATCH_SIZE} max_new_tokens=${MAX_NEW_TOKENS} dtr_max_tokens=${DTR_MAX_TOKENS}"
}

apply_training_profile() {
  local vram_gb
  vram_gb="$(detect_vram_gb)"

  if (( vram_gb >= 30 )); then
    set_default_if_unset MAX_SEQ_LEN "1024"
    set_default_if_unset MICRO_BATCH "2"
    set_default_if_unset GRAD_ACCUM "8"
    set_default_if_unset LORA_R "16"
    set_default_if_unset LORA_ALPHA "32"
    set_default_if_unset GRADIENT_CHECKPOINTING "0"
  elif (( vram_gb >= 20 )); then
    set_default_if_unset MAX_SEQ_LEN "1024"
    set_default_if_unset MICRO_BATCH "2"
    set_default_if_unset GRAD_ACCUM "8"
    set_default_if_unset LORA_R "16"
    set_default_if_unset LORA_ALPHA "32"
    set_default_if_unset GRADIENT_CHECKPOINTING "1"
  elif (( vram_gb >= 14 )); then
    set_default_if_unset MAX_SEQ_LEN "768"
    set_default_if_unset MICRO_BATCH "2"
    set_default_if_unset GRAD_ACCUM "8"
    set_default_if_unset LORA_R "8"
    set_default_if_unset LORA_ALPHA "16"
    set_default_if_unset GRADIENT_CHECKPOINTING "1"
  elif (( vram_gb >= 10 )); then
    set_default_if_unset MAX_SEQ_LEN "640"
    set_default_if_unset MICRO_BATCH "1"
    set_default_if_unset GRAD_ACCUM "16"
    set_default_if_unset LORA_R "8"
    set_default_if_unset LORA_ALPHA "16"
    set_default_if_unset GRADIENT_CHECKPOINTING "1"
  else
    set_default_if_unset MAX_SEQ_LEN "512"
    set_default_if_unset MICRO_BATCH "1"
    set_default_if_unset GRAD_ACCUM "24"
    set_default_if_unset LORA_R "8"
    set_default_if_unset LORA_ALPHA "16"
    set_default_if_unset GRADIENT_CHECKPOINTING "1"
  fi

  # Disable eval during training by default for speed.
  set_default_if_unset EVAL_RATIO "0.0"
  echo "[auto-profile] training vram_gb=${vram_gb} max_seq_len=${MAX_SEQ_LEN} micro_batch=${MICRO_BATCH} grad_accum=${GRAD_ACCUM} lora_r=${LORA_R} grad_ckpt=${GRADIENT_CHECKPOINTING}"
}
