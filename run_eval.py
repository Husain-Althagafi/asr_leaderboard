from models.whisper import run_whisper
from eval import normalize_arabic_text, calculate_wer
import os
import time

timing = int(time.time())
model_id = 'openai/whisper-large-v3'
data_manifest = 'C:/Users/husain_althagafi/work/leaderboard_asr/datasets/commonvoice_test.json'

data_folders = [
    'horrid-qvc/CommonVoice18Test',
    'horrid-qvc/Sada22Test',
    'horrid-qvc/MBB2Test',
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


for data_folder in data_folders:
    os.makedirs(f'outputs/{data_folder}', exist_ok=True)
    output_manifest = f'outputs/{data_folder}/{timing}.txt'

    run_whisper(
        model_id=model_id,
        data_manifest=data_manifest,
        data_folder=data_folder,
        output_manifest=output_manifest
    )

    results = calculate_wer(output_manifest)

    with open(results_file, 'a') as f:
        f.write(f'model: {model_id.split('/')[1]}\ndataset: {data_folder.split('/')[1]}\nwer: {results[0]}\ncer: {results[1]}\n\n')
