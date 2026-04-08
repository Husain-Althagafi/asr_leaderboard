#!/usr/bin/bash/env bash



python run_eval.py --run_inference --model /mnt/d/storage/models/whisper-large-v3 --model_type whisper --output_manifest baseline_whisper --random_sample

python run_eval.py --run_inference --model /mnt/d/storage/models/whisper-large-v3 --model_type whisper --lora_model /mnt/d/storage/models/whisper_lora_ar_pseudo --output_manifest finetuned_whisper --random_sample

python run_eval.py --run_inference --model /mnt/d/storage/models/davids --model_type faster-whisper --output_manifest davids --random_sample

python run_eval.py --run_inference --model /mnt/d/storage/models/qwen-asr-1.7b --model_type qwen-asr --output_manifest qwen --random_sample
