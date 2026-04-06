from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

class WhisperLoraWrapper:
    def __init__(self, model_path, lora_path, device, dtype=torch.float16):
        self.processor = AutoProcessor.from_pretrained(model_path)
        base_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True)
        self.model = PeftModel.from_pretrained(base_model, lora_path)

        self.model.to(device)
        self.model.eval()


    def __call__(self, audio_array, sr, language='ar'):
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features.to(self.model.device, dtype=self.model.dtype)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")

        with torch.inference_mode():
            generated_ids = self.model.generate(inputs, max_length=448, num_beams=5, forced_decoder_ids=forced_decoder_ids)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


