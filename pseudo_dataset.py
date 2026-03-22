from datasets import load_from_disk
import json
from pathlib import Path


def main():
    ds = load_from_disk(f'D:/DATA/ArabicVoicesClean_v4')['train']

    filepath = Path(f'gemini_labels.jsonl')

    mapping = {}
    with filepath.open('r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line)

            mapping[content['index']] = content['gemini_transcript']

    transcripts = [
        mapping.get(i, None)
        for i in range(len(ds))
    ]

    ds = ds.add_column('gemini_transcript', transcripts)

    ds.save_to_disk('D:/DATA/ArabicVoicesPseudo_v4')

    ds.push_to_hub(f'horrid-qvc/ArabicVoicesPseudo_v4')

if __name__ == '__main__':
    main()