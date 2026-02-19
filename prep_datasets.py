from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk, Audio
import soundfile as sf
import os
import json

def standardize_dataset(ds):
    cols = ds.column_names

    # Find audio column
    if "audio" not in cols:
        raise ValueError(f"No 'audio' column found. Columns: {cols}")

    # Find text column
    text_col = None
    for c in ["text", "sentence", "transcription", "transcript", "normalized_text"]:
        if c in cols:
            text_col = c
            break

    if text_col is None:
        raise ValueError(f"No transcript column found. Columns: {cols}")

    # Keep only needed columns
    ds = ds.select_columns(["audio", text_col])

    # Rename transcript column to "text"
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    return ds


def arrow_to_wav(data_path):
    ds = load_from_disk(data_path)

    # Make sure audio column is decoded
    ds = ds.cast_column("audio", Audio())

    out_dir = "data/{}"
    os.makedirs(out_dir, exist_ok=True)

    manifest_path = "data/common_manifest.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            uid = ex.get("id") or f"utt_{i}"
            uid = str(uid)

            wav_path = os.path.join(out_dir, f"{uid}.wav")

            audio = ex["audio"]
            sf.write(wav_path, audio["array"], audio["sampling_rate"])

            row = {
                "id": uid,
                "wav": wav_path,
                "text": ex["text"]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Export complete.")


if __name__ == "__main__":
    common = load_dataset('horrid-qvc/CommonVoice18Test')['test']
    uae = load_dataset('horrid-qvc/CasablancaUAETest')['test']
    morocco = load_dataset('horrid-qvc/CasablancaMoroccoTest')['test']
    jordan = load_dataset('horrid-qvc/CasablancaJordanTest')['test']
    algeria = load_dataset('horrid-qvc/CasablancaAlgeriaTest')['test']
    sada = load_dataset('horrid-qvc/Sada22Test')['test']
    yemen = load_dataset('horrid-qvc/CasablancaYemenTest')['test']
    palestine = load_dataset('horrid-qvc/CasablancaPalestineTest')['test']  
    mauritania = load_dataset('horrid-qvc/CasablancaMauritaniaTest')['test']
    egypt = load_dataset('horrid-qvc/CasablancaEgyptTest')['test']
    mgb2 = load_dataset('horrid-qvc/MGB2Test')['test']
    
    common = standardize_dataset(common)
    uae = standardize_dataset(uae)
    morocco = standardize_dataset(morocco)
    jordan = standardize_dataset(jordan)
    algeria = standardize_dataset(algeria)
    sada = standardize_dataset(sada)
    yemen = standardize_dataset(yemen)
    palestine = standardize_dataset(palestine)
    mauritania = standardize_dataset(mauritania)
    egypt = standardize_dataset(egypt)
    mgb2 = standardize_dataset(mgb2)

    combined = concatenate_datasets([common, uae, morocco, jordan, algeria, sada, yemen, palestine, mauritania, egypt, mgb2])
    
    save_path = 'data'
    os.makedirs(save_path, exist_ok=True)
    common.save_to_disk(f'{save_path}/common')
    uae.save_to_disk(f'{save_path}/uae')
    morocco.save_to_disk(f'{save_path}/morocco')
    jordan.save_to_disk(f'{save_path}/jordan')  
    algeria.save_to_disk(f'{save_path}/algeria')
    sada.save_to_disk(f'{save_path}/sada')
    yemen.save_to_disk(f'{save_path}/yemen')
    palestine.save_to_disk(f'{save_path}/palestine')
    mauritania.save_to_disk(f'{save_path}/mauritania')  
    egypt.save_to_disk(f'{save_path}/egypt')
    mgb2.save_to_disk(f'{save_path}/mgb2')
