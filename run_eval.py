from models.whisper import run_whisper
from eval import normalize_arabic_text, calculate_wer
import os
import time


model_id = 'openai/whisper-large-v3'
data_manifest = 'C:/Users/husain_althagafi/work/leaderboard_asr/datasets/commonvoice_test.json'
data_folder = 'horrid-qvc/Sada22Test'
os.makedirs(f'outputs/{data_folder}', exist_ok=True)
output_manifest = f'outputs/{data_folder}/{time.time()}'

run_whisper(
    model_id=model_id,
    data_manifest=data_manifest,
    data_folder=data_folder,
    output_manifest=output_manifest
)

results = calculate_wer(output_manifest)

