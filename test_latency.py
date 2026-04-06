import argparse
import os
from wrappers.WhisperLoraWrapper import WhisperLoraWrapper
from wrappers.WhisperTurboWrapper import WhisperTurboWrapper
from wrappers.QwenASRWrapper import QwenASRWrapper
from datasets import load_from_disk
import time

MODEL_REGISTRY = {
    'whisper-turbo': lambda args: WhisperTurboWrapper(args.model, device=args.device),
    'whisper-lora': lambda args: WhisperLoraWrapper(args.model, args.lora_path, device=args.device),
    'qwen-asr': lambda args: QwenASRWrapper(args.model, device=args.device),
}


def parse_args():
    parser = argparse.ArgumentParser(description='Test latency of a model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--lora_path', type=str, required=False, help='Path to the LoRA adapter file')
    parser.add_argument('--model_type', type=str, required=True, choices=MODEL_REGISTRY.keys(), help='Type of the model to use')
    parser.add_argument('--samples', type=int, default=10, help='Number of runs to average latency')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--results_path', type=str, required=True, help='Path to the latency tests results file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print(f'Building model wrapper for {args.model_type}...')
    model = MODEL_REGISTRY[args.model_type](args)
    print('Model wrapper built. Running latency test...')
    print(f'Loading audio samples...')

    ds = load_from_disk('data/sada')
    ds = ds.select(range(args.samples + args.warmup))
    print(f'Audio samples loaded. Running latency test on {args.samples} samples with {args.warmup} warmup runs...')

    os.makedirs(f'outputs/latency-tests', exist_ok=True)
    with open(f'outputs/latency-tests/{args.results_path}.txt', 'w') as f:
        f.write(f'Latency test for model: {args.model}\n')
        f.flush()
        # Warmup runs
        print(f'Running {args.warmup} warmup runs...\n\n')
        for i in range(args.warmup):
            sample = ds.select([i])[0]
            before = time.time()
            model(sample['audio']['array'], sample['audio']['sampling_rate']) 
            delta = time.time() - before 

            duration_sec = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]

            f.write(f'Warmup run {i + 1}/{args.warmup}. Time: {delta:.4f} seconds. Audio duration: {duration_sec:.4f} seconds\n')
            f.flush()
            print(f'Warmup run {i + 1} latency: {delta:.4f} seconds')

        # Timing runs
        total_time = 0.0
        total_audio_duration = 0.0
        rtf_total = 0.0

        print(f'\nWarmup complete. Running {args.samples} timed runs...\n\n')
        for i in range(args.samples): 
            sample = ds.select([args.warmup + i])[0]
            before = time.time()
            model(sample['audio']['array'], sample['audio']['sampling_rate'])  
            delta = time.time() - before
            total_time += delta

            duration_sec = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]
            total_audio_duration += duration_sec

            rtf = duration_sec / delta
            rtf_total += rtf

            f.write(f'Timing run {i + 1}/{args.samples}. Time: {delta:.4f} seconds. Audio duration: {duration_sec:.4f} seconds. Real-Time Factor: {rtf:.4f}\n')
            f.flush()
            print(f'Run {i + 1} latency: {delta:.4f} seconds')


        average_latency = total_time / args.samples
        average_rtf = rtf_total / args.samples
        f.write(f'Average latency over {args.samples} runs: {average_latency:.4f} seconds\nThroughput: {total_audio_duration / total_time:.4f} seconds of audio per second\nAverage Real-Time Factor: {average_rtf:.4f}\n')
        f.flush()
        print(f'Average latency over {args.samples} runs: {average_latency:.4f} seconds')



