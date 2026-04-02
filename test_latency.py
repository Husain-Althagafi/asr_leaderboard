import argparse
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from wrappers.WhisperLoraWrapper import WhisperLoraWrapper



def parse_args():
    parser = argparse.ArgumentParser(description='Test latency of a model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--lora_path', type=str, required=True, help='Path to the LoRA adapter file')
    parser.add_argument('--samples', type=int, default=10, help='Number of runs to average latency')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()