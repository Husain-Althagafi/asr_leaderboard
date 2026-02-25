from tqdm import tqdm
from eval import normalize_arabic_text
import json
import argparse
import os
import Levenshtein

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ASR models on multiple datasets')

    parser.add_argument(
        '--output_manifest',
        type=str,
        help='Path to output manifest of the model inference to evaluate',
    )

    return parser.parse_args()


def read_manifest(datafolder):
    predictions = []
    target_transcripts = []

    with open(datafolder, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            target_transcripts.append(normalize_arabic_text(item['text']))
            predictions.append(normalize_arabic_text(item['pred_text']))

    return predictions, target_transcripts


def top_5_distances(predictions, target_transcripts, dfs):
    top_5_distances = []

    for pred, target, datafolder in zip(predictions, target_transcripts, dfs):
        distance = Levenshtein.distance(pred, target)
        if distance > top_5_distances[-1][0] if top_5_distances else True:
            top_5_distances.append((distance, pred, target, datafolder))
            top_5_distances.sort(key=lambda x:x[0], reverse=True)
            if len(top_5_distances) > 5:
                top_5_distances.pop()

    for distance, pred, target, datafolder in top_5_distances:
        print(f"\nDatafolder: {datafolder}\nDistance: {distance}\nPrediction: {pred}\nTarget: {target}\n\n------------------------------")


if __name__ == '__main__':
    args = parse_args()

    output_manifest = args.output_manifest
    datafolders = os.listdir(output_manifest)

    predictions = []
    target_transcripts = []
    dfs = []

    for datafolder in datafolders:
        p, t = read_manifest(output_manifest + f'/{datafolder}')
        predictions.extend(p)
        target_transcripts.extend(t)
        dfs.extend([datafolder] * len(p))

    top_5_distances(predictions, target_transcripts, dfs)
    