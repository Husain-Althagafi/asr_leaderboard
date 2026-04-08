#!/usr/bin/bash/env bash



python run_eval.py --run_inference --model D:/storage/models/whisper-large-v3 --model_type whisper --output_manifest baseline_newcodebase1 --random_sample

python run_eval.py --run_inference --model D:/storage/models/whisper-large-v3 --model_type whisper --lora_model D:/storage/models/whisper_lora_ar_pseudo --output_manifest finetuned_newcodebase1 --random_sample

python run_eval.py --run_inference --model D:/storage/models/davids --model_type faster-whisper --output_manifest davids_newcodebase1 --random_sample

python run_eval.py --run_inference --model D:/storage/models/qwen-asr-1.7b --model_type qwen-asr --output_manifest qwen_newcodebase --random_sample
