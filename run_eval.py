from transformers import AutoModelForSpeechSeq2Seq
import torch
from models.whisper import run_whisper
from eval import calculate_wer
import os
import time
from pydub import AudioSegment
import argparse

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

args = parser.parse_args()

model_id = args.model
# model_id = 'D:/storage/whisper-large-v3'

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
results_file = f'outputs/final_results/{timing}.txt'

wer_total = 0
cer_total = 0
count = 0

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )

total_len_ds = 0

for data_folder in data_folders:
    print(f'\n\n-------------------------------------------------------------')
    print(f'Running evaluation for dataset: {data_folder.split("/")[1]}')
    print(f'-------------------------------------------------------------\n\n')

    os.makedirs(f'outputs/{timing}', exist_ok=True)
    output_manifest = f'outputs/{timing}/{data_folder.split("/")[1]}.txt'
    # output_manifest = f'outputs/run_outputs/{data_folder.split("/")[1]}.txt'

    len_ds = run_whisper(
        model_id=model_id,
        data_folder=data_folder,
        output_manifest=output_manifest,
        model=model
    )

    total_len_ds += len_ds

    results = calculate_wer(output_manifest)
    wer_total += results[0]
    cer_total += results[1]
    count += 1

    with open(results_file, 'a', encoding = 'utf-8') as f:
        f.write(f'model: whisper-large-v3\ndataset: {data_folder.split("/")[1]}\nwer: {results[0]}\ncer: {results[1]}\nlen_ds: {len_ds}\n\n')
        # f.write(f'model: whisper-large-v3\ndataset: {data_folder.split("/")[1]}\nwer: {results[0]}\ncer: {results[1]}\n\n')

with open(results_file, 'a', encoding='utf-8') as f:
    f.write(f'-------------------average wer: {wer_total/count} average cer: {cer_total/count}-------------------\n\n')