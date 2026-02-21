import json
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from torch.utils.data import Dataloader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def run_whisper(model_id, data_manifest, data_folder, output_manifest):
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
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
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
        torch_dtype=torch_dtype,
        device=device,
    )


    ds = load_dataset(data_folder)['test']
    loader = Dataloader(ds, batch_size=1, shuffle=False)
    
    with open(output_manifest, 'w') as fout:
            all_inference_time = 0
            all_audio_duration = 0
            all_inference_memory = []
            count = 0
            for item in tqdm(loader):
                audio = item["audio"]
                text = item['text']
                duration = item["duration"]

                torch.cuda.reset_max_memory_allocated(torch.device("cuda"))
                initial_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)

                start_time = time.time()
                transcription = pipe(audio, generate_kwargs = {"language":"<|ar|>","task": "transcribe"})["text"]
                end_time = time.time()
                
                inference_time = end_time - start_time
                if count > 4:
                    all_inference_time += inference_time
                    all_audio_duration += duration
                peak_memory = torch.cuda.max_memory_allocated(torch.device("cuda"))/(1024 ** 3)
                all_inference_memory.append(peak_memory-initial_memory)        
                count += 1

                metadata = {
                    "text": item['text'],
                    "pred_text": transcription,
                }
                json.dump(metadata, fout, ensure_ascii=False)
                fout.write('\n')

    print("average rtf : ", all_inference_time/all_audio_duration)
    print("model memory : ", initial_memory)
    print("average inference-only memory : ", sum(all_inference_memory)/len(all_inference_memory))
