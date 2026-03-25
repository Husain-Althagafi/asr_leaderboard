from datasets import load_from_disk, Audio
from google import genai
from google.genai import types

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from pathlib import Path
import json
import threading
import time


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--testing', action='store_true')

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

    parser.add_argument(
        '--num_workers',
        type=int,
        default=3
    )

    parser.add_argument(
        '--max_output_tokens',
        type=int,
        default=128
    )

    parser.add_argument(
        '--project',
        type=str,
        default='gen-lang-client-0040506510'
    )

    parser.add_argument(
        '--location',
        type=str,
        default='us-central1'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash'
    )

    parser.add_argument(
        '--max_retries',
        type=int,
        default=3
    )

    parser.add_argument(
        '--retry_sleep',
        type=float,
        default=2.0
    )

    return parser.parse_args()


thread_local = threading.local()


def get_client(project: str, location: str):
    if not hasattr(thread_local, "client"):
        thread_local.client = genai.Client(
            vertexai=True,
            project=project,
            location=location
        )
    return thread_local.client


def process_sample(i, sample, args):
    client = get_client(args.project, args.location)

    sample_bytes = sample['audio']['bytes']
    if sample_bytes is None:
        raise ValueError(f"Sample {i} has no audio bytes.")

    last_error = None

    for attempt in range(args.max_retries):
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=[
                    'Transcribe this Arabic audio. Return only the transcript text.',
                    types.Part.from_bytes(data=sample_bytes, mime_type='audio/wav')
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=args.max_output_tokens,
                    temperature=0.0
                )
            )

            transcript = response.text.strip() if response.text else ""

            return {
                'index': i,
                'audio_filepath': sample.get('audio_filepath'),
                'original_transcript': sample.get('text'),
                'gemini_transcript': transcript
            }

        except Exception as e:
            last_error = e
            if attempt < args.max_retries - 1:
                time.sleep(args.retry_sleep * (2 ** attempt))

    return {
        'index': i,
        'audio_filepath': sample.get('audio_filepath'),
        'original_transcript': sample.get('text'),
        'error': str(last_error)
    }


def load_processed_indices(output_path: Path):
    processed = set()

    if not output_path.exists():
        return processed

    with output_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                idx = record.get('index')
                if idx is not None:
                    processed.add(idx)
            except Exception:
                pass

    return processed


def main():
    args = parse_args()

    ds = load_from_disk(args.data_path)['train']
    print(f'Dataset loaded: {ds}')

    if args.testing:
        ds = ds.select(range(min(args.samples, len(ds))))

    ds = ds.cast_column('audio', Audio(decode=False))

    print(f'Length of dataset to be pseudo-labeled: {len(ds)}')

    output_path = Path(args.output_path)

    processed = set()
    if args.resume:
        processed = load_processed_indices(output_path)
        print(f'Found {len(processed)} already processed samples.')

    samples_to_process = [
        (i, sample)
        for i, sample in enumerate(ds)
        if i not in processed
    ]

    print(f'Samples remaining: {len(samples_to_process)}')

    if len(samples_to_process) == 0:
        print('Nothing to do.')
        return

    with output_path.open('a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_sample, i, sample, args): i
                for i, sample in samples_to_process
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                i = futures[future]

                try:
                    record = future.result()
                except Exception as e:
                    record = {
                        'index': i,
                        'audio_filepath': None,
                        'original_transcript': None,
                        'error': str(e)
                    }

                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()

    print('Done.')


if __name__ == '__main__':
    main()