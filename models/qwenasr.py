import json
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Audio
import numpy as np
from pydub import AudioSegment
from pydub.utils import which
import os
import io

from qwen_asr import Qwen3ASRModel


device = "cuda:0" if torch.cuda.is_available() else "cpu"
np.random.seed(42)

FFMPEG_BIN = r"C:\Users\husain_althagafi\Downloads\ffmpeg\ffmpeg\bin"
os.environ["PATH"] = FFMPEG_BIN + ";" + os.environ["PATH"]

AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

print("ffmpeg:", which("ffmpeg"))
print("ffprobe:", which("ffprobe"))
print("AudioSegment.converter:", AudioSegment.converter)
print("AudioSegment.ffprobe:", AudioSegment.ffprobe)


def load_audio_from_bytes(blob: bytes):
    try:
        seg = AudioSegment.from_file(io.BytesIO(blob))
    except Exception as e1:
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

    if seg.channels == 2:
        x = x.reshape((-1, 2)).mean(axis=1)

    max_val = float(1 << (8 * seg.sample_width - 1))
    x = x.astype(np.float32) / max_val

    return x, sr


class QwenASRWrapper:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation='flash_attention_2',
            max_inference_batch_size=1,
            max_new_tokens=4096,
        )

    def __call__(self, audio_array, sr, language="Arabic"):
        return self.model.transcribe((audio_array, sr), language=language)[0].text


def run_qwen_asr(
    model_id,
    data_folder,
    output_manifest,
    model=None,
    full=False,
    random=False,
    proportional=False,
    language="Arabic",
):
    """
    Arguments
    ---------
    model_id: str
        Path or HF name for the Qwen ASR model.
    data_folder: str
        Dataset name or local dataset folder.
    output_manifest: str
        Output JSONL manifest path.
    model:
        Optional preloaded QwenASRWrapper.
    full: bool
        If True, evaluate on full dataset.
    random: bool
        If True, sample randomly.
    proportional: bool
        If True and full=False, sample 10% instead of 100 samples.
    language: str
        Language passed to Qwen ASR.

    Output
    ------
    Creates an output manifest containing ground truths and predictions.
    """

    if model is None:
        model = QwenASRWrapper(model_id, device=device)

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

            transcription = model(audio, sr, language=language)

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