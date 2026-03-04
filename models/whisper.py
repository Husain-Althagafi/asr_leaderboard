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

FFMPEG_BIN = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN + ";" + os.environ["PATH"]

AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe  = os.path.join(FFMPEG_BIN, "ffprobe.exe")

# optional sanity prints
print("ffmpeg:", which("ffmpeg"))
print("ffprobe:", which("ffprobe"))
print("AudioSegment.converter:", AudioSegment.converter)
print("AudioSegment.ffprobe:", AudioSegment.ffprobe)


def load_audio_from_bytes(blob: bytes):
    # 1) Let ffmpeg sniff the container/codec (most robust)
    try:
        seg = AudioSegment.from_file(io.BytesIO(blob))
    except Exception as e1:
        # 2) If sniffing fails, try common formats explicitly
        for fmt in ("wav", "mp3", "flac", "ogg", "m4a", "webm"):
            try:
                seg = AudioSegment.from_file(io.BytesIO(blob), format=fmt)
                break
            except Exception:
                seg = None
        if seg is None:
            raise RuntimeError(f"Could not decode audio bytes. First error: {e1}")

    sr = int(seg.frame_rate)

    x = np.array(seg.get_array_of_samples())

    # handle stereo
    if seg.channels == 2:
        x = x.reshape((-1, 2)).mean(axis=1)

    # normalize based on sample width (more correct than /32768 always)
    # sample_width is bytes per sample: 2 -> int16, 4 -> int32, etc.
    max_val = float(1 << (8 * seg.sample_width - 1))
    x = x.astype(np.float32) / max_val

    return x, sr


def run_whisper(model_id, data_folder, output_manifest, model=None):
    """
    Arguments
    ---------
    model_id: str
        HuggingFace whisper model name
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
    if model is None:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

    else:
        model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        dtype=torch_dtype,
        device=device,
    )

    ds = load_dataset(data_folder)['test'] if 'CasablancaAllTest' not in data_folder else load_from_disk(f'd:/storage/{data_folder}')
    ds = ds.cast_column("audio", Audio(decode=False))
    random_indices = np.random.choice(len(ds), size=100, replace=False)
    ds = ds.select(random_indices)
    len_ds = len(ds)
    print(f'Loaded {len_ds} samples from the dataset.') 
    
    with open(output_manifest, 'w', encoding='utf-8') as fout:
            all_inference_memory = []
            count = 0
            for item in tqdm(ds):
                path = item["audio"]["path"] 
                audio, sr = load_audio_from_bytes(item["audio"]["bytes"])

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                transcription = pipe(
                    {"array": audio, "sampling_rate": int(sr)},
                    generate_kwargs={"language":"<|ar|>", "task":"transcribe", 'max_length': None}
                )["text"]
                                
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                all_inference_memory.append(peak_memory-initial_memory)        
                count += 1

                metadata = {
                    "text": item['text'],
                    'path': path,
                    "pred_text": transcription,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))

    return len_ds