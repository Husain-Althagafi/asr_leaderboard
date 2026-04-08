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


def run_faster_whisper(
    model_id,
    data_folder,
    output_manifest,
    model,
    full=False,
    random=False,
    proportional=False,
):
    """
    Arguments
    ---------
    model_id: str
        Kept for interface consistency; not used if `model` is already loaded.
    data_folder: str
        The dataset name or local dataset folder.
    output_manifest: str
        The output manifest path.
    model:
        A Faster-Whisper model instance.
    full: bool
        If True, run on full dataset.
    random: bool
        If True, sample randomly.
    proportional: bool
        If True and full=False, sample 10% of dataset instead of 100.

    Output
    ------
    Creates an output manifest containing ground truths and predictions.
    """

    if 'CasablancaAllTest' in data_folder:
        ds = load_from_disk(f'C:/Users/husain_althagafi/work/leaderboard_asr/data/{data_folder}')
    elif 'ArabicVoicesClean_v4' in data_folder:
        ds = load_from_disk(f'D:/{data_folder}')['train']
    else:
        ds = load_dataset(data_folder)['test']

    ds = ds.cast_column("audio", Audio(decode=False))

    print(f'full sampling: {full}')

    if not full:
        sample_size = int(0.1 * len(ds)) if proportional else 100
        sample_size = min(sample_size, len(ds))

        selected_indices = (
            np.random.choice(len(ds), size=sample_size, replace=False)
            if random else list(range(sample_size))
        )
        ds = ds.select(selected_indices)
        original_indices = list(selected_indices)
    else:
        original_indices = list(range(len(ds)))

    len_ds = len(ds)
    print(f'Loaded {len_ds} samples from the dataset.')

    all_inference_memory = []

    with open(output_manifest, 'w', encoding='utf-8') as fout:
        for idx, item in enumerate(tqdm(ds)):
            original_idx = int(original_indices[idx])
            path = item["audio"]["path"]
            audio, sr = load_audio_from_bytes(item["audio"]["bytes"])

            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = (
                    torch.cuda.max_memory_allocated(torch.device("cuda")) / (1024 ** 3)
                )
            else:
                initial_memory = 0.0

            segments, info = model.transcribe(
                audio,
                language="ar",
            )

            transcription = " ".join(seg.text.strip() for seg in segments).strip()

            if torch.cuda.is_available():
                peak_memory = (
                    torch.cuda.max_memory_allocated(torch.device("cuda")) / (1024 ** 3)
                )
                all_inference_memory.append(peak_memory - initial_memory)

            metadata = {
                "index": original_idx,
                "text": item["text"],
                "path": path,
                "pred_text": transcription,
            }
            json.dump(metadata, fout, ensure_ascii=False)
            fout.write('\n')

    print("model memory :", initial_memory)
    if all_inference_memory:
        print("average inference-only memory :", sum(all_inference_memory) / len(all_inference_memory))
    else:
        print("average inference-only memory : 0.0")

    return len_ds