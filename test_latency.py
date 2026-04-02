import argparse
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from wrappers.WhisperLoraWrapper import WhisperLoraWrapper


def build_whisper_turbo_wrapper(args):
    return WhisperTurboWrapper(args.model, device=args.device)

def build_whisper_lora_wrapper(args):
    return WhisperLoraWrapper(args.model, args.lora_path, args.device)

MODEL_REGISTRY = {
    'whisper-turbo': build_whisper_turbo_wrapper,
    'whisper-lora': build_whisper_lora_wrapper,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Test latency of a model')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--lora_path', type=str, required=False, help='Path to the LoRA adapter file')
    parser.add_argument('--model_type', type=str, required=True, choices=MODEL_REGISTRY.keys(), help='Type of the model to use')
    parser.add_argument('--samples', type=int, default=10, help='Number of runs to average latency')
    parser.add_argument('--warmup', type=int, default=3, help='Number of warmup runs')
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    print(f'Building model wrapper for {args.model_type}...')
    model = MODEL_REGISTRY[args.model_type](args)
    print('Model wrapper built. Running latency test...')
    print(f'Loading audio samples...')

    ds = load_dataset('horrid-qvc/Sada22Test', split='test').cast_column("audio", Audio(decode=False))
    ds = ds.select(range(args.samples + args.warmup))
    print(f'Audio samples loaded. Running latency test on {args.samples} samples with {args.warmup} warmup runs...')

    # Warmup runs
    for i in range(args.warmup):
        print(f'Warmup run {i + 1}/{args.warmup}...')
        for sample in ds[:args.warmup]:
            model(sample['audio']['path'])  

    # Timing runs
    import time
    total_time = 0.0
    for i in range(args.samples): 
        print(f'Timing run {i + 1}/{args.samples}...')
        start_time = time.time()
        for sample in ds[args.warmup:args.warmup + args.samples]:
            model(sample['audio']['path'])  
        end_time = time.time()
        run_time = end_time - start_time
        total_time += run_time
        print(f'Run {i + 1} latency: {run_time:.4f} seconds')
    average_latency = total_time / args.samples
    print(f'Average latency over {args.samples} runs: {average_latency:.4f} seconds')

    

