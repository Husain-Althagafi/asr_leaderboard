from datasets import load_dataset, Audio
from eval import normalize_arabic_text
from tqdm import tqdm


def combine_labels(sample):
    w = normalize_arabic_text(sample['whisper_pseudolabel']).split()
    t = normalize_arabic_text(sample['text']).split()

    if len(t) < len(w):
        try:
            idx = w.index(t[-1])
        except ValueError:
            idx = -1

        if idx >= 0:
            p = t + w[idx + 1:] if len(w) > idx else t
            return " ".join(p)
        else:
            return " ".join(t)

    return " ".join(t)


if __name__ == '__main__':
    print('Loading dataset...')
    ds = load_dataset('horrid-qvc/ArabicVoices_WhisperPseudos')['train']
    ds = ds.cast_column('audio', Audio(decode=False))
    print('Dataset loaded.')

    combined_labels_list = []
    for sample in tqdm(ds):
        combined_label = combine_labels(sample)
        combined_labels_list.append(combined_label)

    print(len(combined_labels_list))

    ds = ds.add_column('combined_label', combined_labels_list)

    ds.push_to_hub('horrid-qvc/ArabicVoices_Combinedv4')
    print('Pushed to hub.')