from transformers import AutoModelForSpeechSeq2Seq
import torch
from models.whisper import run_whisper
from models.faster_whisper import run_faster_whisper
from eval import calculate_wer
import os
import time
from pydub import AudioSegment
import argparse
from peft import PeftModel
from faster_whisper import WhisperModel


FFMPEG = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffprobe.exe"
AudioSegment.converter = FFMPEG
AudioSegment.ffprobe = FFPROBE

FFMPEG = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffprobe.exe"
AudioSegment.converter = FFMPEG
AudioSegment.ffprobe = FFPROBE

timing = int(time.time())

parser = argparse.ArgumentParser(description='Evaluate ASR models on multiple datasets')

parser.add_argument(
    '--model',
    type=str,
    default='D:/storage/whisper-large-v3',
    help='Path or name for the ASR model to evaluate',
)

parser.add_argument(
    '--lora_model',
    type=str,
    default=None,
    help='Path to lora adapters to use for evaluation (if any)',
)

parser.add_argument(
    '--output_manifest',
    type=str,
    default=None,
    help='Path to output manifest file',
)

parser.add_argument(
    '--run_inference',
    action='store_true',
)

parser.add_argument(
    '--faster_whisper',
    action='store_true',
)

args = parser.parse_args()

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = args.model
# model_id = 'D:/storage/whisper-large-v3'

if args.lora_model is not None and args.run_inference:
    print("loading base model for lora...")
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    ) if args.faster_whisper == False else WhisperModel(model_id, device="cuda", compute_type="float16")
    print("loading lora model...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
    )  

elif args.run_inference:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                ) if args.faster_whisper == False else WhisperModel(model_id, device="cuda", compute_type="float16")
        model.to("cuda" if torch.cuda.is_available() else "cpu")

data_folders = [
    'horrid-qvc/CommonVoice18Test',
    'horrid-qvc/Sada22Test',
    'horrid-qvc/MGB2Test',
    'data/horrid-qvc/CasablancaAllTest',
    # 'horrid-qvc/CasablancaUAETest',
    # 'horrid-qvc/CasablancaMoroccoTest',
    # 'horrid-qvc/CasablancaJordanTest',
    # 'horrid-qvc/CasablancaAlgeriaTest',
    # 'horrid-qvc/CasablancaYemenTest',
    # 'horrid-qvc/CasablancaPalestineTest',
    # 'horrid-qvc/CasablancaMauritaniaTest',
    # 'horrid-qvc/CasablancaEgyptTest',
]

os.makedirs(f'outputs/final_results', exist_ok=True)
results_file = f'outputs/final_results/{timing}.txt' if args.output_manifest is None else f'outputs/final_results/{args.output_manifest}.txt'

wer_total = 0
cer_total = 0
count = 0

total_len_ds = 0

error_rates = []

for data_folder in data_folders:
    print(f'\n\n-------------------------------------------------------------')
    print(f'Running evaluation for dataset: {data_folder.split("/")[1]}')
    print(f'-------------------------------------------------------------\n\n')

    os.makedirs(f'outputs/{timing}', exist_ok=True) if args.output_manifest is None else os.makedirs(args.output_manifest, exist_ok=True)
    output_manifest = f'outputs/{timing}/{data_folder.split("/")[1]}.txt' if args.output_manifest is None else f'outputs/'+args.output_manifest+f'/{data_folder.split("/")[1]}.txt'
    # output_manifest = f'outputs/run_outputs/{data_folder.split("/")[1]}.txt'

    if args.run_inference:
        run_whisper(
            model_id=model_id,
            data_folder=data_folder,
            output_manifest=output_manifest,
            model=model
        ) if args.faster_whisper == False else run_faster_whisper(
            model_id=model_id,
            data_folder=data_folder,
            output_manifest=output_manifest,
            model=model
        )

    results = calculate_wer(output_manifest)
    wer_total += results[0]
    cer_total += results[1]
    len_ds = results[2]
    count += 1
    total_len_ds += len_ds

    error_rates.append((data_folder.split("/")[1], results[0], results[1], len_ds))

    with open(results_file, 'a', encoding = 'utf-8') as f:
        f.write(f'model: whisper-large-v3\ndataset: {data_folder.split("/")[1]}\nwer: {results[0]}\ncer: {results[1]}\nlen_ds: {len_ds}\n\n')

for err in error_rates:
    wer_total += err[1] * err[3] if err[3] > 0 else 0
    cer_total += err[2] * err[3] if err[3] > 0 else 0

with open(results_file, 'a', encoding='utf-8') as f:
    f.write(f'-------------------average wer: {wer_total/total_len_ds if total_len_ds > 0 else 0} average cer: {cer_total/total_len_ds if total_len_ds > 0 else 0}-------------------\n\n')