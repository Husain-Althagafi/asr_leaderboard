from models.whisper import run_whisper
from eval import normalize_arabic_text, calculate_wer
import os
import time

data_manifest = '/c/Users/husain_althagafi/work/leaderboard_asr/datasets/datamanifest'
data_folder = '/c/Users/husain_althagafi/work/leaderboard_asr/datasets'
output_manifest = f'/c/Users/husain_althagafi/work/leaderboard_asr/outputs/{time.time()}.txt'

run_whisper(
    model_id='C:/Users/husain_althagafi/work/storage/whisper-large-v3',
    data_manifest=data_manifest,
    data_folder=data_folder,
    output_manifest=output_manifest
)


