from transformers import AutoModelForSpeechSeq2Seq
import torch
from models.whisper import run_whisper
from eval import normalize_arabic_text, calculate_wer
import os
import time
from pydub import AudioSegment

FFMPEG = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffprobe.exe"
AudioSegment.converter = FFMPEG
AudioSegment.ffprobe = FFPROBE

FFMPEG = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin\ffprobe.exe"
AudioSegment.converter = FFMPEG
AudioSegment.ffprobe = FFPROBE

timing = int(time.time())
# model_id = 'openai/whisper-large-v3'
model_id = 'D:/storage/whisper-large-v3'
data_manifest = 'C:/Users/husain_althagafi/work/leaderboard_asr/datasets/commonvoice_test.json'

data_folders = [
    # 'horrid-qvc/CommonVoice18Test',
    # 'horrid-qvc/Sada22Test',
    'horrid-qvc/MGB2Test',
    'horrid-qvc/CasablancaUAETest',
    'horrid-qvc/CasablancaMoroccoTest',
    'horrid-qvc/CasablancaJordanTest',
    'horrid-qvc/CasablancaAlgeriaTest',
    'horrid-qvc/CasablancaYemenTest',
    'horrid-qvc/CasablancaPalestineTest',
    'horrid-qvc/CasablancaMauritaniaTest',
    'horrid-qvc/CasablancaEgyptTest',
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

for data_folder in data_folders:
    print(f'\n\n-------------------------------------------------------------')
    print(f'Running evaluation for dataset: {data_folder.split("/")[1]}')
    print(f'-------------------------------------------------------------\n\n')

    os.makedirs(f'outputs/{timing}', exist_ok=True)
    output_manifest = f'outputs/{timing}/{data_folder.split("/")[1]}.txt'

    run_whisper(
        model_id=model_id,
        data_manifest=data_manifest,
        data_folder=data_folder,
        output_manifest=output_manifest,
        model=model
    )

    results = calculate_wer(output_manifest)
    wer_total += results[0]
    cer_total += results[1]
    count += 1

    with open(results_file, 'a', encoding = 'utf-8') as f:
        f.write(f'model: {'whisper-large-v3'}\ndataset: {data_folder.split("/")[1]}\nwer: {results[0]}\ncer: {results[1]}\n\n')

with open(results_file, 'a', encoding='utf-8') as f:
    f.write(f'-------------------averagewer: {wer_total/count} average cer: {cer_total/count}-------------------\n\n')