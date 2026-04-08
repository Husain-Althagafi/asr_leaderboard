import json
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, load_from_disk
from datasets import Audio
import numpy as np
from pydub import AudioSegment
from pydub.utils import which
import os
import soundfile as sf
import io

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
np.random.seed(42)


def full_eval(data_folder, output_manifest, model, full=False, random=False, proportional=False):
    """
    Arguments
    ---------
    model_id: str
        HuggingFace whisper model name
    processor: AutoProcessor
        The processor corresponding to the model
    data_manifest: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_manifest: str
        The output manifest path

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    

    if 'ArabicVoicesClean_v4' in data_folder:
        ds = load_from_disk(f'/mnt/d/storage/{data_folder}')['train']
    
    else:
        ds = load_from_disk(f'/home/husain_althagafi/work/asr_leaderboard/data/{data_folder}')

    ds = ds.cast_column("audio", Audio(decode=True))

    print(f'full sampling: {full}')
    print(f'sampling rate: {ds[0]["audio"]["sampling_rate"]}')

    if not full:
        sample_size = int(0.1 * len(ds)) if proportional else 100
        selected_indices = np.random.choice(len(ds), size=sample_size, replace=False) if random else list(range(sample_size))
        ds = ds.select(selected_indices)
        original_indices = selected_indices
    else:
        original_indices = list(range(len(ds)))        

    len_ds = len(ds)
    print(f'Loaded {len_ds} samples from the dataset.') 
    
    with open(output_manifest, 'w', encoding='utf-8') as fout:
            all_inference_memory = []
            count = 0
            for idx, item in enumerate(tqdm(ds)):
                original_idx = int(original_indices[idx])
                path = item["audio"]["path"] 
                audio, sr = item['audio']['array'], item['audio']['sampling_rate']

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                transcription = model(audio, sr)
                                
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                all_inference_memory.append(peak_memory-initial_memory)        
                count += 1

                metadata = {
                    "index": original_idx,
                    "text": item['text'],
                    'path': path,
                    "pred_text": transcription,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))

    return len_ds