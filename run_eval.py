from transformers import AutoModelForSpeechSeq2Seq
import torch
from wrappers.QwenASRWrapper import QwenASRWrapper
from wrappers.WhisperLoraWrapper import WhisperLoraWrapper
from wrappers.WhisperTurboWrapper import WhisperTurboWrapper
from eval import calculate_wer
import os
import time
import argparse
from peft import PeftModel
from unified_eval import full_eval

timing = int(time.time())

parser = argparse.ArgumentParser(description='Evaluate ASR models on multiple datasets')

parser.add_argument(
    '--model',
    type=str,
    default='/mnt/d/storage/models/whisper-large-v3',
    help='Path or name for the ASR model to evaluate',
)

parser.add_argument(
    '--model_type',
    type=str,
    required=True,
    choices=['whisper', 'faster-whisper', 'qwen-asr'],
    help='Type of model backend to evaluate',
)

parser.add_argument(
    '--lora_model',
    type=str,
    default=None,
    help='Path to lora adapters to use for evaluation (Whisper only)',
)

parser.add_argument(
    '--output_manifest',
    type=str,
    default=None,
    help='Path prefix to output manifests/results',
)

parser.add_argument(
    '--run_inference',
    action='store_true',
)

parser.add_argument(
     '--sample_proportion',
     action='store_true',
)

parser.add_argument(
    '--full_eval',
    action='store_true',
)

parser.add_argument(
    '--random_sample',
    action='store_true',
)

args = parser.parse_args()

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = args.model

if args.output_manifest == None:
    print(f'WARNING NO OUTPUT MANIFEST HAS BEEN SPECIFIED. NO RESULTS WILL BE SAVED. To save results, specify --output_manifest with a name for the output file (e.g. --output_manifest mymodel_results)')


def load_model():
    if args.model_type == 'qwen-asr':
        if args.lora_model is not None:
            raise ValueError('--lora_model is only supported for whisper, not qwen-asr.')
        print("loading qwen asr model...")
        model = QwenASRWrapper(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        return model

    if args.model_type == 'faster-whisper':
        if args.lora_model is not None:
            raise ValueError('--lora_model is not supported with faster-whisper in this script.')
        print("loading faster-whisper model...")
        model = WhisperTurboWrapper(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        return model 
    
    # standard whisper
    else:
        print("loading base whisper model for lora...")
        model =  WhisperLoraWrapper(
            model_id,
            args.lora_model,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.model.eval()
        return model
    

model = load_model() if args.run_inference else None

data_folders = [
    'common',
    # 'horrid-qvc/Sada22Test',
    # 'horrid-qvc/MGB2Test',
    # 'horrid-qvc/CasablancaUAETest',
    # 'horrid-qvc/CasablancaMoroccoTest',
    # 'horrid-qvc/CasablancaJordanTest',
    # 'horrid-qvc/CasablancaAlgeriaTest',
    # 'horrid-qvc/CasablancaYemenTest',
    # 'horrid-qvc/CasablancaPalestineTest',
    # 'horrid-qvc/CasablancaMauritaniaTest',
    # 'horrid-qvc/CasablancaEgyptTest',
    # 'storage/ArabicVoicesClean_v4',
    # 'horrid-qvc/CasablancaAllTest',
]

os.makedirs('outputs/final_results', exist_ok=True)
results_file = (
    f'outputs/final_results/{timing}.txt'
    if args.output_manifest is None
    else f'outputs/final_results/{args.output_manifest}.txt'
)

weighted_wer_total = 0.0
weighted_cer_total = 0.0
total_len_ds = 0
error_rates = []

if args.run_inference:
    print(f'\n\nRunning inference and evaluation for model: {args.model} of type {args.model_type}\n\n')

for data_folder in data_folders:
    dataset_name = data_folder.split("/")[-1]

    print(f'\n\n-------------------------------------------------------------')
    print(f'Running evaluation for dataset: {dataset_name}')
    print(f'-------------------------------------------------------------\n\n')

    out_dir = f'outputs/{timing}' if args.output_manifest is None else f'outputs/{args.output_manifest}'
    os.makedirs(out_dir, exist_ok=True)

    output_manifest = f'{out_dir}/{dataset_name}.txt'

    if args.run_inference:
        full_eval(
            data_folder=data_folder,
            output_manifest=output_manifest,
            model=model,
            full=args.full_eval,
            random=args.random_sample,
            proportional=args.sample_proportion,
        )
        

    results = calculate_wer(output_manifest)
    wer, cer, len_ds = results[0], results[1], results[2]

    total_len_ds += len_ds
    weighted_wer_total += wer * len_ds if len_ds > 0 else 0
    weighted_cer_total += cer * len_ds if len_ds > 0 else 0

    error_rates.append((dataset_name, wer, cer, len_ds))

    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(
            f'model_type: {args.model_type}\n',
            f'lora_model: {args.lora_model}\n' if args.lora_model is not None else 'None\n',
            f'model: {args.model}\n'
            f'dataset: {dataset_name}\n'
            f'wer: {wer}\n'
            f'cer: {cer}\n'
            f'len_ds: {len_ds}\n\n'
        )

avg_wer = weighted_wer_total / total_len_ds if total_len_ds > 0 else 0
avg_cer = weighted_cer_total / total_len_ds if total_len_ds > 0 else 0

with open(results_file, 'a', encoding='utf-8') as f:
    f.write(
        f'-------------------'
        f'average wer: {avg_wer} '
        f'average cer: {avg_cer}'
        f'-------------------\n\n'
    )