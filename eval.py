from tqdm import tqdm
import json
# from nemo.collections.asr.metrics.wer import word_error_rate
from tqdm import tqdm
import re
from jiwer import wer, cer
import unicodedata

def remove_punctuation(text):
    return ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != 'P'
    )

def normalize_arabic_text(text):
    """
    Arabic text normalization:
    1. Remove punctuation
    2. Remove diacritics
    3. Eastern Arabic numerals to Western Arabic numerals

    Arguments
    ---------
    text: str
        text to normalize
    Output
    ---------
    normalized text
    """
    # Remove punctuation
    punctuation = r'[!"#$%&\'()*.+,-./:;<=>?@[\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, '', text)

    # Remove diacritics
    diacritics = r'[\u064B-\u0652]'  # Arabic diacritical marks (Fatha, Damma, etc.)
    text = re.sub(diacritics, '', text)
    
    # Normalize Hamzas and Maddas
    text = re.sub('پ', 'ب', text)
    text = re.sub('ڤ', 'ف', text)
    text = re.sub(r'[آ]', 'ا', text)
    text = re.sub(r'[أإ]', 'ا', text)
    text = re.sub(r'[ؤ]', 'و', text)
    text = re.sub(r'[ئ]', 'ي', text)
    text = re.sub(r'[ء]', '', text)   

    text = remove_punctuation(text)

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', 
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    return text.strip()


def calculate_wer(output_manifest):
    """
    Arguments
    ---------
    output_manifest: str
        path to the output manifest of the model inference

    Output
    ---------
    WER/CER
    """
    predictions = []
    target_transcripts = []
    with open(output_manifest, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            target_transcripts.append(normalize_arabic_text(item['text']))
            predictions.append(normalize_arabic_text(item['pred_text']))

    len_ds = [len(predictions)] * len(predictions)
    # wer = word_error_rate(predictions, target_transcripts)
    wer = jiwer.wer(predictions, target_transcripts)

    print("wer : ", wer)
    # cer = word_error_rate(predictions, target_transcripts, use_cer=True)
    cer = jiwer.cer(predictions, target_transcripts,)

    print("cer : ", cer)
    return wer, cer, len_ds[0]

