#!/usr/bin/bash/env bash


python test_latency.py --model /mnt/d/storage/models/qwen-asr-1.7b --model_type qwen-asr --results_path qwen-asr --warmup 5 --samples 100
python test_latency.py --model /mnt/d/storage/models/davids --model_type whisper-turbo --results_path whisper-turbo --warmup 5 --samples 100
python test_latency.py --model /mnt/d/storage/models/whisper-large-v3 --lora_path /mnt/d/storage/models/whisper_lora_ar_pseudo --model_type whisper-lora --results_path whisper-lora --warmup 5 --samples 100