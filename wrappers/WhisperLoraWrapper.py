import argparse
from peft import PeftModel
from safetensors import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import torch

class WhisperLoraWrapper:
    def __init__(self, model_path, lora_path, device, dtype=torch.float16):
        self.processor = AutoProcessor.from_pretrained(model_path)
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True)
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        self.model.to(device)
        self.model.eval()


    def __call__(self, audio_path, language='ar'):
        audio_array, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(self.model.device, dtype=self.model.dtype)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")

        with torch.inference_mode():
            generated_ids = self.model.generate(inputs, max_length=448, num_beams=5, forced_decoder_ids=forced_decoder_ids)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Whisper Lora Wrapper")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base Whisper model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA weights")
    parser.add_argument('--model_type', type=str, required=True, choices=['whisper-large', 'whisper-lora', 'whisper-turbo'], help='Type of the model to use')
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()

    