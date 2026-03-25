from datasets import load_from_disk
from tqdm import tqdm
import json


ds = load_from_disk(f'D:/storage/ArabicVoicesClean_v4')['train']

labels = []
with open('outputs/pseudolabeling_whisper/ArabicVoicesClean_v4.txt', encoding='utf-8') as f:
    for line in tqdm(f):
        labels.append(json.loads(line)['pred_text'])


if len(labels) != len(ds):
    raise ValueError(f'len ds: {len(ds)} != len labels: {len(labels)}')


ds = ds.add_column('whisper_pseudolabel', labels)

ds.save_to_disk('D:/storage/ArabicVoices_WhisperPseudos')
ds.push_to_hub('horrid-qvc/ArabicVoices_WhisperPseudos')