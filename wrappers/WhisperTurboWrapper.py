import argparse
from faster_whisper import WhisperModel


class WhisperTurboWrapper:
    def __init__(self, model_path, device='cuda'):
        self.model = WhisperModel(model_path, device=device, compute_type="float16")

    
    def __call__(self, audio_array, sr, language='ar'):
        segments, _ = self.model.transcribe(audio_array, language=language, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the WhisperTurboWrapper')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the Whisper model')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file')
    args = parser.parse_args()

    print(f'Loading whisper-turbo model from {args.model_path}...')
    wrapper = WhisperTurboWrapper(args.model_path)
    print(f'Processing audio file {args.audio_path}...')
    transcription = wrapper(args.audio_path)
    print(f'transcription: {transcription}')