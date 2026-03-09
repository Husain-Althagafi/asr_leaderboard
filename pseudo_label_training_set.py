from datasets import load_from_disk, Audio
import soundfile as sf
import sounddevice as sd
import io
from google import genai
from google.genai import types

from tqdm import tqdm
import argparse
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--testing',
        action='store_true'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='/home/husain/data/ArabicVoicesClean_v4'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=10
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='gemini_labels.jsonl'
    )

    parser.add_argument(
        '--resume',
        action='store_true'
    )

    return parser.parse_args()



def main():
    args = parse_args()

    client = genai.Client()

    ds = load_from_disk(args.data_path)['train']

    print(f'Dataset shape: {ds}')

    if args.testing:
        ds = ds.select(range(args.samples))

    ds = ds.cast_column('audio', Audio(decode=False))

    print(f'Length of dataset to be pseudo-labeled is: {len(ds)}')

    output_path = Path(args.output_path)

    processed = set()
    if args.resume:
        if output_path.exists():
            with output_path.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        processed.add(json.loads(line)['index'])
                    except:
                        pass

    with output_path.open('a', encoding='utf-8') as f:
        for i, sample in tqdm(enumerate(ds)):
            if i in processed:
                continue

            sample_bytes = sample['audio']['bytes']

            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=[
                    'Transcribe this audio exactly. It is in arabic. Return ONLY the transcription. Dont separate different speakers keep the trasncription continuous.',
                    types.Part.from_bytes(data=sample_bytes, mime_type='audio/wav')
                ]
            )

            record = {
                'index': i,
                'audio_filepath': sample['audio_filepath'],
                'original_transcript': sample['text'],
                'gemini_transcript': response.text
            }

            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()


if __name__ == '__main__':
    main()