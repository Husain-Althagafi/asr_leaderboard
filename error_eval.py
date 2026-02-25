from tqdm import tqdm
from eval import normalize_arabic_text
import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ASR models on multiple datasets')

    parser.add_argument(
        '--output_manifest',
        type=str,
        help='Path to output manifest of the model inference to evaluate',
    )

    return parser.parse_args()


def read_manifest(output_manifest):
    predictions = []
    target_transcripts = []
    with open(output_manifest, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            target_transcripts.append(normalize_arabic_text(item['text']))
            predictions.append(normalize_arabic_text(item['pred_text']))

    return predictions, target_transcripts




if __name__ == '__main__':
    args = parse_args()

    output_manifest = args.output_manifest
    datafolders = os.listdir(output_manifest)

    predictions = []
    target_transcripts = []

    for datafolder in datafolders:
        p, t = read_manifest(output_manifest + f'/{datafolder}')
        predictions.extend(p)
        target_transcripts.extend(t)

    print(len(predictions), len(target_transcripts))