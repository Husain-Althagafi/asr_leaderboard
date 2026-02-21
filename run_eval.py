from models.whisper import run_whisper
from eval import normalize_arabic_text, calculate_wer
import os
import time

model_id = 'C:/Users/husain_althagafi/work/storage/whisper-large-v3'
data_manifest = 'C:/Users/husain_althagafi/work/leaderboard_asr/datasets/commonvoice_test.json'
data_folder = 'C:/Users/husain_althagafi/work/leaderboard_asr/data/common'
output_manifest = f'outputs/test/{time.time()}.txt'

run_whisper(
    model_id=model_id,
    data_manifest=data_manifest,
    data_folder=data_folder,
    output_manifest=output_manifest
)

results = calculate_wer(output_manifest)
