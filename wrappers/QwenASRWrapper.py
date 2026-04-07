from qwen_asr import Qwen3ASRModel
import torch


class QwenASRWrapper:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation='flash_attention_2',
            max_inference_batch_size=1,
            max_new_tokens=4096,
        )

    def __call__(self, audio_array, sr, language='Arabic'):
        return self.model.transcribe((audio_array, sr), language=language)[0].text




    
    
