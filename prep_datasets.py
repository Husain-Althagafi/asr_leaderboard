from datasets import Dataset, concatenate_datasets, load_dataset

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


if __name__ == "__main__":
    common = load_dataset('horrid-qvc/CommonVoice18Test')
    uae = load_dataset('horrid-qvc/CasablancaUAETest')
    morocco = load_dataset('horrid-qvc/CasablancaMoroccoTest')
    jordan = load_dataset('horrid-qvc/CasablancaJordanTest')
    algeria = load_dataset('horrid-qvc/CasablancaAlgeriaTest')
    sada = load_dataset('horrid-qvc/Sada22Test')
    yemen = load_dataset('horrid-qvc/CasablancaYemenTest')
    palestine = load_dataset('horrid-qvc/CasablancaPalestineTest')
    mauritania = load_dataset('horrid-qvc/CasablancaMauritaniaTest')
    egypt = load_dataset('horrid-qvc/CasablancaEgyptTest')
    mgb2 = load_dataset('horrid-qvc/MGB2Test')
    
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
    combined.save_to_disk('CombinedTest')
